// server.js - NVIDIA NIM Proxy - Manual Model Selection (Simple)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA CONFIGURATION =====
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '500';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '16';
process.env.NIM_MAX_CPU_LORAS = process.env.NIM_MAX_CPU_LORAS || '10';

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// ===== MODEL MAPPING - FULL LIST =====
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
  return MODEL_MAPPING[openaiModel] || 'deepseek-ai/deepseek-v4-flash';
}

// ===== RATE LIMITING =====
const MIN_DELAY = 3000; // 3 seconds between requests
let lastRequestTime = 0;
let activeRequests = 0;

async function rateLimit() {
  while (activeRequests >= 1) {
    await sleep(500);
  }
  
  const now = Date.now();
  const wait = MIN_DELAY - (now - lastRequestTime);
  if (wait > 0) {
    console.log(`⏳ Cooldown: ${Math.round(wait/1000)}s...`);
    await sleep(wait);
  }
  
  lastRequestTime = Date.now();
  activeRequests++;
}

// ===== HELPER FUNCTIONS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ===== DETAILED ERROR LOGGING =====
let last429Error = null;
let lastServerError = null;

// ===== RETRY CONFIGURATION =====
const MAX_RETRIES = 5; // 5 retries on server errors for all models

async function callWithRetry(fn, nimModel) {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await fn();
      if (attempt > 0) {
        console.log(`✅ ${nimModel}: Attempt ${attempt + 1}/${MAX_RETRIES + 1} succeeded`);
      }
      return result;
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      
      // ===== 429 HANDLING: NO RETRIES, JUST LOG =====
      if (status === 429) {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🚫 429 RATE LIMIT on ${nimModel}`);
        console.log(`${'='.repeat(60)}`);
        
        if (error.response?.data) {
          console.log(`📦 Body:`, JSON.stringify(error.response.data, null, 2));
        }
        
        if (error.response?.headers) {
          console.log(`📋 Headers:`);
          Object.entries(error.response.headers).forEach(([k, v]) => {
            if (k.startsWith('x-') || k.includes('rate') || k.includes('retry')) {
              console.log(`   ${k}: ${v}`);
            }
          });
        }
        
        last429Error = {
          timestamp: new Date().toISOString(),
          model: nimModel,
          data: error.response?.data,
          headers: error.response?.headers
        };
        
        console.log(`🚫 NO RETRIES on 429`);
        console.log(`${'='.repeat(60)}\n`);
        throw error;
      }
      
      // ===== 500, 503, 504 HANDLING: 5 RETRIES =====
      const isRetryable = status === 500 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        if (isRetryable) {
          console.log(`❌ ${nimModel}: All ${MAX_RETRIES + 1} attempts failed (${status}).`);
        }
        
        lastServerError = {
          timestamp: new Date().toISOString(),
          model: nimModel,
          status,
          attempts: attempt + 1,
          data: error.response?.data
        };
        
        throw error;
      }
      
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
      } else {
        waitTime = 3000 * Math.pow(2, attempt);
        waitTime = waitTime * (0.8 + Math.random() * 0.4);
      }
      
      waitTime = Math.min(waitTime, 30000);
      
      console.log(`⚠️ Server error (${status}) on ${nimModel} - attempt ${attempt + 1}/${MAX_RETRIES + 1} failed. Retrying in ${Math.round(waitTime/1000)}s...`);
      
      if (error.response?.data) {
        console.log(`   Details:`, JSON.stringify(error.response.data).substring(0, 200));
      }
      
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    min_delay: `${MIN_DELAY / 1000}s`,
    active_requests: activeRequests,
    max_retries: MAX_RETRIES,
    retry_on_429: false,
    retry_on_server_errors: '500, 503, 504',
    last_429_error: last429Error,
    last_server_error: lastServerError,
    available_models: Object.keys(MODEL_MAPPING)
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
    
    // Resolve the model YOU chose
    const nimModel = resolveModel(model);
    
    console.log(`📤 ${messages.length} messages → ${nimModel.split('/').pop()} (You selected: ${model})`);
    
    // Wait for rate limit
    await rateLimit();
    
    // NO TOKEN LIMITS - NO TRUNCATION - NO BLOCKING
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
      nimModel
    );
    
    activeRequests--;
    console.log(`✅ Success with ${nimModel.split('/').pop()}`);
    
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
  console.log(`🚀 NIM Proxy - Manual Model Selection`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`📋 Available models:`);
  for (const [openai, nim] of Object.entries(MODEL_MAPPING)) {
    console.log(`   ${openai} → ${nim.split('/').pop()}`);
  }
  console.log(`⏱️ Min delay: ${MIN_DELAY / 1000}s between requests`);
  console.log(`🔄 Max retries: ${MAX_RETRIES} (on 500/503/504)`);
  console.log(`🚫 429: NO RETRIES - Logged and returned`);
  console.log(`📝 Full context - NO truncation, NO limits, NO blocking`);
  console.log(`💡 YOU choose the model manually`);
  console.log(`🔍 Health check: /health`);
});
