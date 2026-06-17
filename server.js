// server.js - NVIDIA NIM Proxy with Detailed 429 Diagnostics
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

// ===== TOKEN-BASED THROTTLING =====
const MAX_TOKENS_PER_SECOND = 50;
const MAX_TOKENS_PER_MINUTE = 2000;
let tokenUsageWindow = [];

// ===== DETAILED 429 LOGGING =====
let last429Error = null;

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
        // CAPTURE DETAILED 429 INFORMATION
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🚫 429 RATE LIMIT ERROR - DETAILED DIAGNOSTICS`);
        console.log(`${'='.repeat(60)}`);
        
        // Response body
        if (error.response?.data) {
          console.log(`📦 Response Body:`);
          console.log(JSON.stringify(error.response.data, null, 2));
        }
        
        // Response headers
        if (error.response?.headers) {
          console.log(`\n📋 Response Headers:`);
          Object.entries(error.response.headers).forEach(([key, value]) => {
            console.log(`   ${key}: ${value}`);
          });
        }
        
        // Analyze the error
        const errorData = error.response?.data;
        const errorHeaders = error.response?.headers || {};
        
        console.log(`\n🔍 Analysis:`);
        
        if (errorData?.message?.toLowerCase().includes('cache')) {
          console.log(`   ⚠️ LoRA cache is full - too many different models active`);
          console.log(`   💡 Fix: Increase NIM_MAX_CPU_LORAS or use fewer model variants`);
        }
        
        if (errorData?.message?.toLowerCase().includes('queue')) {
          console.log(`   ⚠️ Request queue is full - too many pending requests`);
          console.log(`   💡 Fix: Increase NIM_TRITON_MAX_QUEUE_SIZE or slow down requests`);
        }
        
        if (errorData?.message?.toLowerCase().includes('token') || 
            errorData?.message?.toLowerCase().includes('rate')) {
          console.log(`   ⚠️ Token throughput limit exceeded`);
          console.log(`   💡 Fix: Reduce MAX_TOKENS_PER_SECOND or MAX_TOKENS_PER_MINUTE`);
        }
        
        if (errorHeaders['retry-after']) {
          console.log(`   ⏱️ Retry-After: ${errorHeaders['retry-after']} seconds`);
        }
        
        if (errorHeaders['x-ratelimit-remaining']) {
          console.log(`   📊 Rate limit remaining: ${errorHeaders['x-ratelimit-remaining']}`);
          console.log(`   📊 Rate limit total: ${errorHeaders['x-ratelimit-limit']}`);
          console.log(`   📊 Rate limit reset: ${errorHeaders['x-ratelimit-reset']}`);
        }
        
        // Store for health endpoint
        last429Error = {
          timestamp: new Date().toISOString(),
          status: error.response?.status,
          data: error.response?.data,
          headers: error.response?.headers,
          messageCount: context?.messages || 'unknown',
          tokenEstimate: context?.tokens || 'unknown'
        };
        
        console.log(`${'='.repeat(60)}\n`);
        console.log(`🚫 Not retrying - returning 429 to client`);
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
    token_throttle: {
      max_per_second: MAX_TOKENS_PER_SECOND,
      max_per_minute: MAX_TOKENS_PER_MINUTE,
      used_last_minute: tokensLastMinute,
      budget_remaining: Math.max(0, MAX_TOKENS_PER_MINUTE - tokensLastMinute)
    },
    last_429_error: last429Error,
    retry_on_429: false,
    strategy: 'full_diagnostics'
  });
});

// Dedicated 429 diagnostics endpoint
app.get('/debug/429', (req, res) => {
  res.json({
    last_429_error: last429Error,
    current_config: {
      NIM_TRITON_MAX_QUEUE_SIZE: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
      NIM_MAX_CPU_LORAS: process.env.NIM_MAX_CPU_LORAS,
      MAX_TOKENS_PER_SECOND: MAX_TOKENS_PER_SECOND,
      MAX_TOKENS_PER_MINUTE: MAX_TOKENS_PER_MINUTE,
      MIN_DELAY: `${MIN_DELAY/1000}s`
    }
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
    const tokenEstimate = estimateTokens(messages);
    
    console.log(`📤 Request: ${messages.length} messages (~${tokenEstimate} tokens) → ${nimModel}`);
    
    await tokenThrottle(messages);
    await rateLimit();
    
    // Prepare request - FULL CONTEXT, NO MESSAGE MERGING
    const nimRequest = {
      model: nimModel,
      messages: messages,
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
      `${nimModel} (msgs: ${messages.length}, tokens: ~${tokenEstimate})`
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
  console.log(`🚀 NVIDIA NIM Proxy with Full 429 Diagnostics`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🔍 Debug endpoint: http://localhost:${PORT}/debug/429`);
  console.log(`📋 Triton queue: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
  console.log(`💾 Max CPU LoRAs: ${process.env.NIM_MAX_CPU_LORAS}`);
  console.log(`🔢 Concurrent requests: 1`);
  console.log(`⏱️ Min delay: ${MIN_DELAY/1000}s`);
  console.log(`🪙 Token limit: ${MAX_TOKENS_PER_SECOND}/s, ${MAX_TOKENS_PER_MINUTE}/min`);
  console.log(`🔄 Retries: ${MAX_RETRIES} (500/503/504 only - NO retries on 429)`);
  console.log(`📝 Full context preserved - NO message merging`);
  console.log(`💡 Strategy: Token throttle + request pacing + full 429 diagnostics`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
});
