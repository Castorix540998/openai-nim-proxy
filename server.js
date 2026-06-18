// server.js - NVIDIA NIM Proxy with Model Rotation
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA CONFIGURATION =====
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '500';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '16';
process.env.NIM_MAX_CPU_LORAS = process.env.NIM_MAX_CPU_LORAS || '10';

// ===== MODEL ROTATION POOL =====
const MODEL_POOL = [
  'deepseek-ai/deepseek-v4-pro',
  'deepseek-ai/deepseek-v4-flash',
  'z-ai/glm-5.1',
  'moonshotai/kimi-k2.6',
  'minimaxai/minimax-m3',
];

let currentModelIndex = 0;
const modelCooldowns = new Map();
const COOLDOWN_DURATION = 900000; // 15 MINUTES cooldown if locked out

function getNextAvailableModel() {
  const now = Date.now();
  
  // Try each model in the pool
  for (let i = 0; i < MODEL_POOL.length; i++) {
    const model = MODEL_POOL[currentModelIndex];
    const cooldownUntil = modelCooldowns.get(model) || 0;
    
    currentModelIndex = (currentModelIndex + 1) % MODEL_POOL.length;
    
    if (now >= cooldownUntil) {
      return model;
    }
  }
  
  // All models on cooldown - find the one that recovers soonest
  let soonestModel = MODEL_POOL[0];
  let soonestTime = Infinity;
  
  for (const model of MODEL_POOL) {
    const cooldownUntil = modelCooldowns.get(model) || 0;
    if (cooldownUntil < soonestTime) {
      soonestTime = cooldownUntil;
      soonestModel = model;
    }
  }
  
  const waitTime = soonestTime - now;
  console.log(`⚠️ All models on cooldown! ${soonestModel.split('/').pop()} recovers in ${Math.round(waitTime/1000)}s`);
  return soonestModel;
}

function setModelCooldown(model) {
  const cooldownUntil = Date.now() + COOLDOWN_DURATION;
  modelCooldowns.set(model, cooldownUntil);
  const minutes = Math.round(COOLDOWN_DURATION / 60000);
  console.log(`🔒 ${model.split('/').pop()} LOCKED OUT for ${minutes} minutes (until ${new Date(cooldownUntil).toLocaleTimeString()})`);
}

// ===== DETAILED 429 LOGGING =====
let last429Error = null;

// ===== RATE LIMITING =====
const MIN_DELAY = 40000; // 40 SECONDS between requests
let lastRequestTime = 0;
let activeRequests = 0;

async function rateLimit() {
  // Wait if there's already an active request
  while (activeRequests >= 1) {
    console.log(`⏳ Waiting for active request to complete...`);
    await sleep(1000);
  }
  
  // Enforce 40 second minimum delay between requests
  const now = Date.now();
  const wait = MIN_DELAY - (now - lastRequestTime);
  if (wait > 0) {
    console.log(`⏳ Rate limit: Waiting ${Math.round(wait/1000)}s before sending to NVIDIA...`);
    await sleep(wait);
  }
  
  lastRequestTime = Date.now();
  activeRequests++;
}

// ===== RETRY CONFIGURATION =====
const MAX_RETRIES = 2;
const RETRY_BASE_DELAY = 3000;

// ===== HELPERS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function callWithRetry(fn, model) {
  let lastError;
  let currentModel = model;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn(currentModel);
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      
      if (status === 429) {
        // Log detailed diagnostics
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🚫 429 on ${currentModel.split('/').pop()}`);
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
        
        // Lock out this model for 15 MINUTES
        setModelCooldown(currentModel);
        
        if (attempt < MAX_RETRIES) {
          const nextModel = getNextAvailableModel();
          console.log(`🔄 Switching to ${nextModel.split('/').pop()}...`);
          currentModel = nextModel;
          continue;
        }
        
        last429Error = {
          timestamp: new Date().toISOString(),
          model: currentModel,
          data: error.response?.data,
          headers: error.response?.headers
        };
        
        console.log(`${'='.repeat(60)}\n`);
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
      } else {
        waitTime = RETRY_BASE_DELAY * Math.pow(2, attempt);
      }
      
      waitTime = Math.min(waitTime, 30000);
      
      console.log(`⚠️ Server error on ${currentModel.split('/').pop()} - retry ${attempt + 1}/${MAX_RETRIES + 1}. Waiting ${Math.round(waitTime/1000)}s...`);
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Model mapping - routes everything through the rotation pool
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
  // Ignore the requested model - always use rotation pool
  return getNextAvailableModel();
}

// Health check
app.get('/health', (req, res) => {
  const now = Date.now();
  const cooldownRemaining = Math.max(0, MIN_DELAY - (now - lastRequestTime));
  
  const modelStatus = MODEL_POOL.map(model => {
    const cooldownUntil = modelCooldowns.get(model) || 0;
    const remaining = Math.max(0, cooldownUntil - now);
    return {
      model: model.split('/').pop(),
      available: now >= cooldownUntil,
      cooldown_remaining: remaining > 0 ? `${Math.round(remaining / 1000)}s` : 'none'
    };
  });
  
  res.json({
    status: 'ok',
    active_requests: activeRequests,
    request_cooldown: `${Math.round(cooldownRemaining / 1000)}s`,
    min_delay: `${MIN_DELAY / 1000}s`,
    model_pool_size: MODEL_POOL.length,
    cooldown_duration: `${Math.round(COOLDOWN_DURATION / 60000)} minutes`,
    model_status: modelStatus,
    last_429: last429Error,
    strategy: 'model_rotation_15min_cooldown'
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    if (!messages?.length) {
      return res.status(400).json({
        error: { message: 'No messages provided', code: 400 }
      });
    }
    
    // Pick the best available model from rotation pool
    const nimModel = getNextAvailableModel();
    
    console.log(`📤 ${messages.length} messages → ${nimModel.split('/').pop()}`);
    
    // 40 SECOND DELAY before sending
    await rateLimit();
    
    // NO TOKEN LIMITS - NO TRUNCATION - Full context sent as-is
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };
    
    const response = await callWithRetry(
      (modelToUse) => axios.post(`${NIM_API_BASE}/chat/completions`, {
        ...nimRequest,
        model: modelToUse
      }, {
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
    console.error('❌ All models failed:', error.message);
    
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

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, code: 404 } });
});

app.listen(PORT, () => {
  console.log(`🚀 NIM Proxy - Model Rotation (15min Cooldown)`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🔄 Rotating between:`);
  MODEL_POOL.forEach(m => console.log(`   • ${m.split('/').pop()}`));
  console.log(`🔒 Per-model cooldown: 15 MINUTES if 429`);
  console.log(`⏱️ Delay between requests: 40 SECONDS`);
  console.log(`📝 Full context - NO truncation, NO token limits`);
  console.log(`💡 Strategy: Heavy spacing + long cooldowns to avoid lockouts`);
});
