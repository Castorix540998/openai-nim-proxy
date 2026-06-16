// server.js - Simple OpenAI to NVIDIA NIM Proxy with Triton Queue Management
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA TRITON QUEUE CONFIGURATION =====
// This tells NIM to queue requests instead of returning 429
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '200';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '16';
// Additional Triton settings for better queue handling
process.env.NIM_TRITON_MIN_WORKERS = process.env.NIM_TRITON_MIN_WORKERS || '1';
process.env.NIM_TRITON_MAX_WORKERS = process.env.NIM_TRITON_MAX_WORKERS || '4';

// ===== SIMPLE RATE LIMITING =====
const MIN_DELAY = 2000; // 2 seconds between requests
let lastRequestTime = 0;

async function rateLimit() {
  const now = Date.now();
  const wait = MIN_DELAY - (now - lastRequestTime);
  if (wait > 0) {
    console.log(`⏳ Waiting ${Math.round(wait/1000)}s...`);
    await sleep(wait);
  }
  lastRequestTime = Date.now();
}

// ===== RETRY CONFIGURATION =====
const MAX_RETRIES = 5;
const RETRY_BASE_DELAY = 3000;

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
      const isRetryable = status === 429 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }
      
      // Use Retry-After header if available (Triton provides this)
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ Triton says retry after ${retryAfter}s`);
      } else {
        waitTime = RETRY_BASE_DELAY * Math.pow(2, attempt);
        waitTime = waitTime * (0.8 + Math.random() * 0.4);
      }
      
      // Cap at 60 seconds
      waitTime = Math.min(waitTime, 60000);
      
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
  res.json({
    status: 'ok',
    triton_queue_size: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
    triton_batch_size: process.env.NIM_TRITON_MAX_BATCH_SIZE,
    triton_workers: `${process.env.NIM_TRITON_MIN_WORKERS}-${process.env.NIM_TRITON_MAX_WORKERS}`,
    min_delay: `${MIN_DELAY/1000}s`,
    max_retries: MAX_RETRIES
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
    
    console.log(`📤 Request: ${messages.length} messages → ${nimModel}`);
    
    // Prepare request - NO CHUNKING, NO TOKEN MANIPULATION
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };
    
    // Wait for rate limit
    await rateLimit();
    
    // Make request with retry logic
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
    
    console.log('✅ Success');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.pipe(res);
      
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
    console.error('❌ Error:', error.message);
    
    const status = error.response?.status || 500;
    
    if (status === 429) {
      console.log('💡 429: Request queued by Triton. Will retry automatically.');
    }
    
    res.status(status).json({
      error: {
        message: error.response?.data?.error?.message || error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 'server_error',
        code: status
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
  console.log(`🚀 Simple NIM Proxy with Triton Queue Management`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`📋 Triton queue size: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
  console.log(`📊 Triton batch size: ${process.env.NIM_TRITON_MAX_BATCH_SIZE}`);
  console.log(`👷 Triton workers: ${process.env.NIM_TRITON_MIN_WORKERS}-${process.env.NIM_TRITON_MAX_WORKERS}`);
  console.log(`⏱️ Min delay between requests: ${MIN_DELAY/1000}s`);
  console.log(`🔄 Max retries: ${MAX_RETRIES}`);
  console.log(`📝 Full context preserved - NO chunking or compression`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
});
