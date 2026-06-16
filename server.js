// server.js - OpenAI to NVIDIA NIM API Proxy (Optimized with Queue Management)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA TRITON QUEUE CONFIGURATION =====
// This tells NIM to accept more requests in queue instead of returning 429
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '100';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '8';

// ===== CONFIGURATION =====
const MAX_TOKENS_PER_REQUEST = 2000; // Can be slightly larger now
const MAX_RESPONSE_TOKENS = 2048;
const RATE_LIMIT_DELAY = 8000;      // 8 seconds (reduced from 15s)
const MAX_RETRIES = 4;              // More retries since queue handles waiting
const RETRY_DELAY_BASE = 5000;      // 5 second base retry delay

// ===== REQUEST QUEUE MANAGEMENT =====
class RequestQueue {
  constructor() {
    this.queue = [];
    this.processing = false;
  }

  async add(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject });
      this.process();
    });
  }

  async process() {
    if (this.processing || this.queue.length === 0) return;
    
    this.processing = true;
    
    while (this.queue.length > 0) {
      const { fn, resolve, reject } = this.queue.shift();
      
      try {
        // Wait rate limit delay between requests
        const now = Date.now();
        const timeSinceLastRequest = now - (this.lastRequestTime || 0);
        
        if (timeSinceLastRequest < RATE_LIMIT_DELAY) {
          const waitTime = RATE_LIMIT_DELAY - timeSinceLastRequest;
          console.log(`⏳ Queue: Waiting ${Math.round(waitTime/1000)}s before next request...`);
          await sleep(waitTime);
        }
        
        this.lastRequestTime = Date.now();
        const result = await fn();
        resolve(result);
      } catch (error) {
        reject(error);
      }
    }
    
    this.processing = false;
  }
}

const requestQueue = new RequestQueue();

// ===== CONTEXT MANAGEMENT =====
class ContextManager {
  prepareContext(messages) {
    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    const totalTokens = estimateTokens(JSON.stringify(messages));
    
    // If under limit, return as-is
    if (totalTokens <= MAX_TOKENS_PER_REQUEST) {
      return messages;
    }
    
    console.log(`🔄 Compressing ${totalTokens} tokens to fit ${MAX_TOKENS_PER_REQUEST} limit...`);
    
    const preparedMessages = [];
    
    // Keep system message (truncated if needed)
    if (systemMessages.length > 0) {
      preparedMessages.push({
        role: 'system',
        content: this.truncate(systemMessages[0].content, 400)
      });
    }
    
    // Find the last user message index
    let lastUserIndex = -1;
    for (let i = conversationMessages.length - 1; i >= 0; i--) {
      if (conversationMessages[i].role === 'user') {
        lastUserIndex = i;
        break;
      }
    }
    
    if (lastUserIndex >= 0) {
      // Add context from before the last user message
      if (lastUserIndex > 0) {
        const contextMsg = conversationMessages[lastUserIndex - 1];
        preparedMessages.push({
          role: contextMsg.role,
          content: this.truncate(contextMsg.content, 300)
        });
      }
      
      // Add the last user message
      preparedMessages.push({
        role: 'user',
        content: this.truncate(conversationMessages[lastUserIndex].content, 800)
      });
      
      // Add any responses after the last user message
      for (let i = lastUserIndex + 1; i < conversationMessages.length; i++) {
        preparedMessages.push({
          role: conversationMessages[i].role,
          content: this.truncate(conversationMessages[i].content, 500)
        });
      }
    } else {
      // Fallback: last 2 messages
      preparedMessages.push(...conversationMessages.slice(-2).map(msg => ({
        role: msg.role,
        content: this.truncate(msg.content, 400)
      })));
    }
    
    // Create brief context summary for older messages
    if (lastUserIndex > 1) {
      const olderMsgs = conversationMessages.slice(0, lastUserIndex - 1);
      const contextSummary = this.createContextSummary(olderMsgs);
      
      preparedMessages.splice(1, 0, {
        role: 'system',
        content: `[Context: ${contextSummary}]`
      });
    }
    
    const finalTokens = estimateTokens(JSON.stringify(preparedMessages));
    console.log(`✅ Compressed to ${finalTokens} tokens`);
    
    // Emergency: if still too large
    if (finalTokens > MAX_TOKENS_PER_REQUEST) {
      console.log('🚨 Emergency truncation...');
      return [preparedMessages[0], preparedMessages[preparedMessages.length - 1]];
    }
    
    return preparedMessages;
  }
  
  truncate(content, maxTokens) {
    if (!content) return '';
    const maxChars = maxTokens * 4;
    return content.length <= maxChars ? content : content.substring(0, maxChars) + '...';
  }
  
  createContextSummary(messages) {
    // Extract key topics from recent messages
    const topics = messages
      .filter(m => m.role === 'user')
      .slice(-3)
      .map(m => {
        const words = m.content.split(' ').slice(0, 10);
        return words.join(' ');
      })
      .join(' → ');
    
    return `Previous topics: ${topics}`.substring(0, 400);
  }
}

const contextManager = new ContextManager();

// ===== HELPER FUNCTIONS =====

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function estimateTokens(text) {
  if (!text) return 0;
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  return Math.ceil(str.length / 4);
}

async function callWithRetry(fn, context = '') {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      const isRetryable = status === 429 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }
      
      // Check for Retry-After header first
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ Got Retry-After: ${retryAfter}s`);
      } else {
        // Exponential backoff: 5s, 10s, 20s, 40s
        waitTime = RETRY_DELAY_BASE * Math.pow(2, attempt);
        // Add jitter
        waitTime = waitTime * (0.8 + Math.random() * 0.4);
      }
      
      console.log(`⚠️ ${context} attempt ${attempt + 1}/${MAX_RETRIES + 1} failed (${status}). Waiting ${Math.round(waitTime/1000)}s...`);
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

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

function resolveModel(openaiModel) {
  if (MODEL_MAPPING[openaiModel]) return MODEL_MAPPING[openaiModel];
  
  const modelLower = openaiModel.toLowerCase();
  if (modelLower.includes('gpt-4') || modelLower.includes('opus')) {
    return 'meta/llama-3.1-405b-instruct';
  } else if (modelLower.includes('claude') || modelLower.includes('gemini')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    max_tokens: MAX_TOKENS_PER_REQUEST,
    rate_limit_delay: `${RATE_LIMIT_DELAY/1000}s`,
    triton_queue_size: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
    queue_length: requestQueue.queue.length
  });
});

// Models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
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
    
    if (!messages || messages.length === 0) {
      return res.status(400).json({
        error: { message: 'No messages provided', type: 'invalid_request_error', code: 400 }
      });
    }
    
    const nimModel = resolveModel(model);
    
    // Prepare context
    const preparedMessages = contextManager.prepareContext(messages);
    const inputTokens = estimateTokens(JSON.stringify(preparedMessages));
    
    console.log(`📊 Request: ${inputTokens} tokens → ${nimModel}`);
    
    // Prepare request
    const nimRequest = {
      model: nimModel,
      messages: preparedMessages,
      temperature: temperature || 0.6,
      max_tokens: Math.min(max_tokens || MAX_RESPONSE_TOKENS, MAX_RESPONSE_TOKENS),
      stream: stream || false
    };
    
    // Use queue to manage requests
    const response = await requestQueue.add(() =>
      callWithRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 120000 // 2 minute timeout for queued requests
        }),
        nimModel
      )
    );
    
    console.log('✅ Success');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.on('data', (chunk) => res.write(chunk));
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });
    } else {
      const content = response.data.choices[0]?.message?.content || '';
      
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: content },
          finish_reason: response.data.choices[0]?.finish_reason || 'stop'
        }],
        usage: response.data.usage || {
          prompt_tokens: inputTokens,
          completion_tokens: estimateTokens(content),
          total_tokens: inputTokens + estimateTokens(content)
        }
      });
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    
    const status = error.response?.status || 500;
    
    if (status === 429) {
      console.log('💡 Tip: Try increasing NIM_TRITON_MAX_QUEUE_SIZE environment variable');
    }
    
    res.status(status).json({
      error: {
        message: status === 429
          ? 'Server busy. Request queued. Try increasing NIM_TRITON_MAX_QUEUE_SIZE.'
          : error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 'server_error',
        code: status
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Proxy running on port ${PORT}`);
  console.log(`📦 Max input: ${MAX_TOKENS_PER_REQUEST} tokens`);
  console.log(`📝 Max response: ${MAX_RESPONSE_TOKENS} tokens`);
  console.log(`⏱️ Rate: 1 request per ${RATE_LIMIT_DELAY/1000}s`);
  console.log(`📋 Triton queue size: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
  console.log(`🔄 Max retries: ${MAX_RETRIES}`);
});
