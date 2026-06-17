// server.js - NVIDIA NIM Proxy with Message Merging
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA TRITON & LORA CONFIGURATION =====
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '500';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '16';
process.env.NIM_TRITON_MIN_WORKERS = process.env.NIM_TRITON_MIN_WORKERS || '1';
process.env.NIM_TRITON_MAX_WORKERS = process.env.NIM_TRITON_MAX_WORKERS || '4';
process.env.NIM_MAX_CPU_LORAS = process.env.NIM_MAX_CPU_LORAS || '10';

// ===== MESSAGE COUNT LIMIT WORKAROUND =====
const MAX_MESSAGES = 10; // NVIDIA's limit
const MAX_TOKENS_PER_SECOND = 50;
const MAX_TOKENS_PER_MINUTE = 2000;
let tokenUsageWindow = [];

// Merge older messages to stay under the 10 message limit
function mergeMessagesUnderLimit(messages) {
  if (!messages || messages.length <= MAX_MESSAGES) {
    return messages;
  }
  
  console.log(`🔧 Merging ${messages.length} messages to fit ${MAX_MESSAGES} limit...`);
  
  const systemMessages = messages.filter(m => m.role === 'system');
  const nonSystemMessages = messages.filter(m => m.role !== 'system');
  
  // Keep last 8 messages intact (4 exchanges)
  const recentMessages = nonSystemMessages.slice(-8);
  const olderMessages = nonSystemMessages.slice(0, -8);
  
  if (olderMessages.length === 0) {
    // Just truncate to last 10
    return [...systemMessages.slice(0, 1), ...nonSystemMessages.slice(-9)];
  }
  
  // Merge all older messages into a single system message
  const mergedContent = olderMessages
    .map(m => `${m.role}: ${m.content.substring(0, 200)}`)
    .join('\n');
  
  const summaryMessage = {
    role: 'system',
    content: `[Earlier conversation:\n${mergedContent.substring(0, 1000)}\n---\nContinue the conversation naturally.]`
  };
  
  // Combine: 1 system + 1 summary + 8 recent = 10 messages total
  const result = [
    ...systemMessages.slice(0, 1),
    summaryMessage,
    ...recentMessages
  ].slice(0, MAX_MESSAGES);
  
  console.log(`✅ Merged to ${result.length} messages`);
  return result;
}

function estimateTokens(messages) {
  if (!messages) return 0;
  return Math.ceil(JSON.stringify(messages).length / 4);
}

async function tokenThrottle(messages) {
  const estimatedTokens = estimateTokens(messages);
  const now = Date.now();
  
  tokenUsageWindow = tokenUsageWindow.filter(entry => now - entry.timestamp < 60000);
  const tokensUsedLastMinute = tokenUsageWindow.reduce((sum, entry) => sum + entry.tokens, 0);
  const tokensUsedLastSecond = tokenUsageWindow
    .filter(entry => now - entry.timestamp < 1000)
    .reduce((sum, entry) => sum + entry.tokens, 0);
  
  console.log(`📊 Token usage: ${tokensUsedLastMinute}/${MAX_TOKENS_PER_MINUTE} per min, ${tokensUsedLastSecond}/${MAX_TOKENS_PER_SECOND} per sec`);
  
  if (tokensUsedLastSecond + estimatedTokens > MAX_TOKENS_PER_SECOND) {
    const waitTime = 1000;
    console.log(`⏳ Token throttle (tok/s): Waiting ${waitTime}ms...`);
    await sleep(waitTime);
  }
  
  if (tokensUsedLastMinute + estimatedTokens > MAX_TOKENS_PER_MINUTE) {
    const oldestEntry = tokenUsageWindow[0];
    if (oldestEntry) {
      const waitTime = 60000 - (now - oldestEntry.timestamp);
      if (waitTime > 0 && waitTime < 60000) {
        console.log(`⏳ Token throttle (tok/min): Waiting ${Math.round(waitTime/1000)}s...`);
        await sleep(waitTime);
      }
    }
  }
  
  tokenUsageWindow.push({ timestamp: Date.now(), tokens: estimatedTokens });
  if (tokenUsageWindow.length > 1000) {
    tokenUsageWindow = tokenUsageWindow.slice(-500);
  }
}

// ===== CONSERVATIVE RATE LIMITING =====
const MIN_DELAY = 5000;
let lastRequestTime = 0;
let activeRequests = 0;

async function rateLimit() {
  while (activeRequests >= 1) {
    console.log(`⏳ Waiting for active request...`);
    await sleep(1000);
  }
  
  const now = Date.now();
  const wait = MIN_DELAY - (now - lastRequestTime);
  if (wait > 0) {
    console.log(`⏳ Request cooldown: ${Math.round(wait/1000)}s...`);
    await sleep(wait);
  }
  
  lastRequestTime = Date.now();
  activeRequests++;
}

// ===== RETRY CONFIGURATION =====
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 5000;

// ===== HELPERS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function callWithRetry(fn, context = '') {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      
      if (status === 429) {
        console.log(`🚫 429 Rate Limited - Not retrying.`);
        throw error;
      }
      
      const isRetryable = status === 500 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }
      
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ Server says retry after ${retryAfter}s`);
      } else {
        waitTime = RETRY_BASE_DELAY * Math.pow(2, attempt);
        waitTime = waitTime * (0.8 + Math.random() * 0.4);
      }
      
      waitTime = Math.min(waitTime, 30000);
      
      console.log(`⚠️ ${context} - Attempt ${attempt + 1}/${MAX_RETRIES + 1} (${status}). Waiting ${Math.round(waitTime/1000)}s...`);
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4': 'minimaxai/minimax-m3',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'z-ai/glm-5.1',
  'claude-3-sonnet': 'mistralai/mistral-medium-3.5-128b',
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b'
};

function resolveModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  
  const lower = model.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('opus')) {
    return 'meta/llama-3.1-405b-instruct';
  } else if (lower.includes('claude') || lower.includes('gemini')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

// Health check
app.get('/health', (req, res) => {
  const now = Date.now();
  const cooldownRemaining = Math.max(0, MIN_DELAY - (now - lastRequestTime));
  const tokensLastMinute = tokenUsageWindow
    .filter(entry => now - entry.timestamp < 60000)
    .reduce((sum, entry) => sum + entry.tokens, 0);
  
  res.json({
    status: 'ok',
    triton_queue_size: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
    max_cpu_loras: process.env.NIM_MAX_CPU_LORAS,
    min_delay: `${MIN_DELAY/1000}s`,
    cooldown_remaining: `${Math.round(cooldownRemaining/1000)}s`,
    active_requests: activeRequests,
    max_messages: MAX_MESSAGES,
    token_throttle: {
      max_per_second: MAX_TOKENS_PER_SECOND,
      max_per_minute: MAX_TOKENS_PER_MINUTE,
      used_last_minute: tokensLastMinute,
      budget_remaining: Math.max(0, MAX_TOKENS_PER_MINUTE - tokensLastMinute)
    },
    retry_on_429: false,
    strategy: 'message_merging'
  });
});

// Models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Main chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    if (!messages?.length) {
      return res.status(400).json({
        error: { message: 'No messages provided', code: 400 }
      });
    }
    
    const nimModel = resolveModel(model);
    
    // WORKAROUND: Merge messages to stay under NVIDIA's 10 message limit
    const processedMessages = mergeMessagesUnderLimit(messages);
    const tokenEstimate = estimateTokens(processedMessages);
    
    console.log(`📤 Request: ${messages.length}→${processedMessages.length} messages (~${tokenEstimate} tokens) → ${nimModel}`);
    
    await tokenThrottle(processedMessages);
    await rateLimit();
    
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };
    
    const response = await callWithRetry(
      () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 180000
      }),
      nimModel
    );
    
    activeRequests--;
    console.log('✅ Success');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.pipe(res);
      
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        activeRequests = Math.max(0, activeRequests - 1);
        res.end();
      });
    } else {
      const content = response.data.choices[0]?.message?.content || '';
      
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content },
          finish_reason: response.data.choices[0]?.finish_reason || 'stop'
        }],
        usage: response.data.usage || {
          prompt_tokens: tokenEstimate,
          completion_tokens: estimateTokens(content),
          total_tokens: tokenEstimate + estimateTokens(content)
        }
      });
    }
    
  } catch (error) {
    activeRequests = Math.max(0, activeRequests - 1);
    console.error('❌ Error:', error.message);
    
    const status = error.response?.status || 500;
    const errorData = error.response?.data;
    
    if (status === 429) {
      if (errorData?.message?.includes('cache')) {
        console.log('💡 429: LoRA cache full');
      } else if (errorData?.message?.includes('queue')) {
        console.log('💡 429: Queue full');
      } else {
        console.log('💡 429: Token throughput limit');
      }
    }
    
    res.status(status).json({
      error: {
        message: error.response?.data?.error?.message || error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 'server_error',
        code: status,
        retry_after: status === 429 ? 10 : null
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 NVIDIA NIM Proxy with Message Merging`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`📋 Triton queue: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
  console.log(`💾 Max CPU LoRAs: ${process.env.NIM_MAX_CPU_LORAS}`);
  console.log(`🔢 Concurrent requests: 1`);
  console.log(`⏱️ Min delay: ${MIN_DELAY/1000}s`);
  console.log(`📝 Message limit: ${MAX_MESSAGES} (older merged into summary)`);
  console.log(`🪙 Token limit: ${MAX_TOKENS_PER_SECOND}/s, ${MAX_TOKENS_PER_MINUTE}/min`);
  console.log(`🔄 Retries: ${MAX_RETRIES} (500/503/504 only)`);
  console.log(`💡 Strategy: Merge messages + token throttle + request pacing`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
});
