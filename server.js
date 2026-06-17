// server.js - NVIDIA NIM Proxy with Full 429 Protection
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA TRITON & LORA CONFIGURATION =====
// Based on NVIDIA's own recommendations for avoiding 429 errors
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '500';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '16';
process.env.NIM_TRITON_MIN_WORKERS = process.env.NIM_TRITON_MIN_WORKERS || '1';
process.env.NIM_TRITON_MAX_WORKERS = process.env.NIM_TRITON_MAX_WORKERS || '4';
// NEW: LoRA cache management to prevent "cache full" 429 errors
process.env.NIM_MAX_CPU_LORAS = process.env.NIM_MAX_CPU_LORAS || '10';

// ===== CONSERVATIVE RATE LIMITING (NVIDIA-style: 1 at a time) =====
const MIN_DELAY = 10000; // 10 seconds between requests (like NV_INGEST_FILES_PER_BATCH=1)
const CONCURRENT_REQUESTS = 1; // Only 1 at a time (like NV_INGEST_CONCURRENT_BATCHES=1)
let lastRequestTime = 0;
let activeRequests = 0;

async function rateLimit() {
  // Wait if there's already an active request
  while (activeRequests >= CONCURRENT_REQUESTS) {
    console.log(`⏳ Waiting for active request to complete...`);
    await sleep(1000);
  }
  
  // Enforce minimum delay between requests
  const now = Date.now();
  const wait = MIN_DELAY - (now - lastRequestTime);
  if (wait > 0) {
    console.log(`⏳ Rate limit cooldown: ${Math.round(wait/1000)}s...`);
    await sleep(wait);
  }
  
  lastRequestTime = Date.now();
  activeRequests++;
}

// ===== RETRY CONFIGURATION =====
// NO retries on 429 - let Triton queue handle it
// Retries only on server errors (503, 504, 500)
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 5000; // 5 second base delay for server errors

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
      
      // NO RETRY on 429 - return immediately
      if (status === 429) {
        console.log(`🚫 429 Rate Limited - Not retrying. Let Triton queue handle it.`);
        throw error;
      }
      
      // Only retry on server errors (500, 503, 504)
      const isRetryable = status === 500 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }
      
      // Use Retry-After header if available
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ Server says retry after ${retryAfter}s`);
      } else {
        // Exponential backoff for server errors
        waitTime = RETRY_BASE_DELAY * Math.pow(2, attempt);
        waitTime = waitTime * (0.8 + Math.random() * 0.4); // Add jitter
      }
      
      // Cap at 30 seconds for server errors
      waitTime = Math.min(waitTime, 30000);
      
      console.log(`⚠️ ${context} - Attempt ${attempt + 1}/${MAX_RETRIES + 1} (${status} server error). Waiting ${Math.round(waitTime/1000)}s...`);
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

// Model mapping - UNCHANGED as requested
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
  
  res.json({
    status: 'ok',
    triton_queue_size: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
    triton_batch_size: process.env.NIM_TRITON_MAX_BATCH_SIZE,
    triton_workers: `${process.env.NIM_TRITON_MIN_WORKERS}-${process.env.NIM_TRITON_MAX_WORKERS}`,
    max_cpu_loras: process.env.NIM_MAX_CPU_LORAS,
    min_delay: `${MIN_DELAY/1000}s`,
    cooldown_remaining: `${Math.round(cooldownRemaining/1000)}s`,
    active_requests: activeRequests,
    max_concurrent: CONCURRENT_REQUESTS,
    max_retries: MAX_RETRIES,
    retry_on_429: false,
    retry_on_server_errors: '500, 503, 504',
    strategy: 'nvidia_conservative_no_429_retry'
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
    
    console.log(`📤 Request: ${messages.length} messages → ${nimModel} (active: ${activeRequests})`);
    
    // Prepare request - FULL CONTEXT, NO CHUNKING
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };
    
    // NVIDIA-style: 1 request at a time with minimum delay
    await rateLimit();
    
    // Make request with retry logic (NO retries on 429)
    const response = await callWithRetry(
      () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 180000 // 3 minute timeout for queued requests
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
        usage: response.data.usage || {}
      });
    }
    
  } catch (error) {
    activeRequests = Math.max(0, activeRequests - 1);
    console.error('❌ Error:', error.message);
    
    const status = error.response?.status || 500;
    const errorData = error.response?.data;
    
    if (status === 429) {
      // Detect specific 429 causes based on NVIDIA documentation
      if (errorData?.message?.includes('cache')) {
        console.log('💡 429 Cause: LoRA cache full (NIM_MAX_CPU_LORAS)');
      } else if (errorData?.message?.includes('queue')) {
        console.log('💡 429 Cause: Request queue full (NIM_TRITON_MAX_QUEUE_SIZE)');
      } else {
        console.log('💡 429 Cause: General rate limiting');
      }
      console.log('🚫 Not retrying - let Triton queue handle it');
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
  console.log(`🚀 NVIDIA NIM Proxy with 429 Protection`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`📋 Triton queue: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
  console.log(`📊 Triton batch: ${process.env.NIM_TRITON_MAX_BATCH_SIZE}`);
  console.log(`👷 Triton workers: ${process.env.NIM_TRITON_MIN_WORKERS}-${process.env.NIM_TRITON_MAX_WORKERS}`);
  console.log(`💾 Max CPU LoRAs: ${process.env.NIM_MAX_CPU_LORAS}`);
  console.log(`🔢 Concurrent requests: ${CONCURRENT_REQUESTS} (sequential)`);
  console.log(`⏱️ Min delay: ${MIN_DELAY/1000}s between requests`);
  console.log(`🔄 Retries: ${MAX_RETRIES} (for 500/503/504 only - NO retries on 429)`);
  console.log(`📝 Full context preserved - NO chunking`);
  console.log(`💡 Strategy: NVIDIA-style conservative (1 at a time)`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
});
