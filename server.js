// server.js - NVIDIA NIM Proxy - 10 Request Batches with 15min Cooldown
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

// ===== MODEL CONFIGURATION =====
const MODEL_PRO = 'deepseek-ai/deepseek-v4-pro';
const MODEL_FLASH = 'deepseek-ai/deepseek-v4-flash';

// ===== BATCH SYSTEM: 10 requests per model, then 15 minute cooldown =====
const MAX_REQUESTS_PER_BATCH = 10;
const COOLDOWN_MINUTES = 15;
const COOLDOWN_MS = COOLDOWN_MINUTES * 60 * 1000;

// Track each model's usage
const modelState = {
  [MODEL_PRO]: {
    requestsUsed: 0,
    cooldownUntil: 0,
    name: 'DeepSeek V4 Pro'
  },
  [MODEL_FLASH]: {
    requestsUsed: 0,
    cooldownUntil: 0,
    name: 'DeepSeek V4 Flash'
  }
};

// Which model is currently active
let activeModel = MODEL_PRO;

function getActiveModel() {
  const now = Date.now();
  const current = modelState[activeModel];
  
  // Check if current model is on cooldown
  if (now < current.cooldownUntil) {
    // Switch to the other model
    const otherModel = activeModel === MODEL_PRO ? MODEL_FLASH : MODEL_PRO;
    const other = modelState[otherModel];
    
    if (now < other.cooldownUntil) {
      // BOTH models on cooldown - reject
      return null;
    }
    
    // Switch to other model
    activeModel = otherModel;
    console.log(`🔄 Switched to ${other.name} (previous model on cooldown)`);
  }
  
  // Check if current model has reached its limit
  if (current.requestsUsed >= MAX_REQUESTS_PER_BATCH) {
    // Put current model on cooldown
    current.cooldownUntil = now + COOLDOWN_MS;
    current.requestsUsed = 0;
    console.log(`🔒 ${current.name}: 10 requests used - COOLDOWN for ${COOLDOWN_MINUTES} minutes (until ${new Date(current.cooldownUntil).toLocaleTimeString()})`);
    
    // Switch to other model
    const otherModel = activeModel === MODEL_PRO ? MODEL_FLASH : MODEL_PRO;
    const other = modelState[otherModel];
    
    if (now < other.cooldownUntil) {
      // Other model also on cooldown - reject
      return null;
    }
    
    // Reset other model's counter if it was previously on cooldown and is now free
    if (other.requestsUsed >= MAX_REQUESTS_PER_BATCH && now >= other.cooldownUntil) {
      other.requestsUsed = 0;
    }
    
    activeModel = otherModel;
    console.log(`🔄 Switched to ${other.name}`);
  }
  
  return activeModel;
}

function incrementRequestCount(model) {
  modelState[model].requestsUsed++;
  const remaining = MAX_REQUESTS_PER_BATCH - modelState[model].requestsUsed;
  console.log(`📊 ${modelState[model].name}: ${modelState[model].requestsUsed}/${MAX_REQUESTS_PER_BATCH} requests used (${remaining} remaining)`);
}

function forceCooldown(model) {
  modelState[model].cooldownUntil = Date.now() + COOLDOWN_MS;
  modelState[model].requestsUsed = MAX_REQUESTS_PER_BATCH; // Mark as fully used
  console.log(`🔒 ${modelState[model].name} FORCED into ${COOLDOWN_MINUTES} minute cooldown (until ${new Date(modelState[model].cooldownUntil).toLocaleTimeString()})`);
}

// ===== HELPER FUNCTIONS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ===== RETRY CONFIGURATION =====
// NO retries on 429
// Retries allowed on 500, 503, 504
const MAX_RETRIES = 5; // More retries for server errors since they're usually temporary
const RETRY_BASE_DELAY = 3000;

async function callWithRetry(fn, model) {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      
      // ===== 429 HANDLING: NO RETRIES, FORCE COOLDOWN =====
      if (status === 429) {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🚫 429 RATE LIMIT on ${modelState[model]?.name || model}`);
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
        
        // IMMEDIATELY force cooldown - NO retries
        forceCooldown(model);
        console.log(`🚫 NO RETRIES - Model locked out for ${COOLDOWN_MINUTES} minutes`);
        console.log(`${'='.repeat(60)}\n`);
        throw error;
      }
      
      // ===== 500, 503, 504 HANDLING: RETRIES ALLOWED =====
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
        waitTime = waitTime * (0.8 + Math.random() * 0.4); // Add jitter
      }
      
      waitTime = Math.min(waitTime, 30000);
      
      console.log(`⚠️ Server error (${status}) on ${modelState[model]?.name || model} - retry ${attempt + 1}/${MAX_RETRIES + 1}. Waiting ${Math.round(waitTime/1000)}s...`);
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Health check - shows detailed status of both models
app.get('/health', (req, res) => {
  const now = Date.now();
  
  const proState = modelState[MODEL_PRO];
  const flashState = modelState[MODEL_FLASH];
  
  const proCooldownRemaining = Math.max(0, proState.cooldownUntil - now);
  const flashCooldownRemaining = Math.max(0, flashState.cooldownUntil - now);
  
  res.json({
    status: 'ok',
    active_model: modelState[activeModel].name,
    deepseek_v4_pro: {
      requests_used: `${proState.requestsUsed}/${MAX_REQUESTS_PER_BATCH}`,
      requests_remaining: MAX_REQUESTS_PER_BATCH - proState.requestsUsed,
      on_cooldown: now < proState.cooldownUntil,
      cooldown_remaining: proCooldownRemaining > 0 ? `${Math.ceil(proCooldownRemaining / 1000)}s (${Math.ceil(proCooldownRemaining / 60000)} minutes)` : 'none',
      available: now >= proState.cooldownUntil && proState.requestsUsed < MAX_REQUESTS_PER_BATCH
    },
    deepseek_v4_flash: {
      requests_used: `${flashState.requestsUsed}/${MAX_REQUESTS_PER_BATCH}`,
      requests_remaining: MAX_REQUESTS_PER_BATCH - flashState.requestsUsed,
      on_cooldown: now < flashState.cooldownUntil,
      cooldown_remaining: flashCooldownRemaining > 0 ? `${Math.ceil(flashCooldownRemaining / 1000)}s (${Math.ceil(flashCooldownRemaining / 60000)} minutes)` : 'none',
      available: now >= flashState.cooldownUntil && flashState.requestsUsed < MAX_REQUESTS_PER_BATCH
    },
    system: {
      batch_size: MAX_REQUESTS_PER_BATCH,
      cooldown_minutes: COOLDOWN_MINUTES,
      retry_policy: {
        on_429: 'NO RETRIES - Immediate 15 minute cooldown',
        on_500_503_504: `Up to ${MAX_RETRIES} retries with exponential backoff`
      }
    }
  });
});

// Models endpoint
app.get('/v1/models', (req, res) => {
  const models = [
    { id: 'gpt-3.5-turbo', object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' },
    { id: 'gpt-4', object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' },
    { id: 'gpt-4o', object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy' }
  ];
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
    
    // Get the active model (respects 10-request batches and cooldowns)
    const nimModel = getActiveModel();
    
    // If no model is available (both on cooldown)
    if (!nimModel) {
      const now = Date.now();
      const proRemaining = Math.max(0, modelState[MODEL_PRO].cooldownUntil - now);
      const flashRemaining = Math.max(0, modelState[MODEL_FLASH].cooldownUntil - now);
      const minRemaining = Math.min(
        proRemaining || Infinity, 
        flashRemaining || Infinity
      );
      
      const minutesRemaining = Math.ceil(minRemaining / 60000);
      const secondsRemaining = Math.ceil(minRemaining / 1000);
      
      console.log(`🚫 Both models on cooldown! Rejecting request.`);
      
      return res.status(429).json({
        error: {
          message: `All models are currently on cooldown. Please wait ${minutesRemaining} minute(s) (${secondsRemaining} seconds) before making another request.`,
          type: 'rate_limit_error',
          code: 429,
          retry_after: secondsRemaining,
          details: {
            deepseek_v4_pro: proRemaining > 0 ? `${Math.ceil(proRemaining / 1000)}s remaining` : 'available',
            deepseek_v4_flash: flashRemaining > 0 ? `${Math.ceil(flashRemaining / 1000)}s remaining` : 'available'
          }
        }
      });
    }
    
    console.log(`📤 ${messages.length} messages → ${modelState[nimModel].name} (${modelState[nimModel].requestsUsed + 1}/${MAX_REQUESTS_PER_BATCH})`);
    
    // Increment the request counter BEFORE making the request
    incrementRequestCount(nimModel);
    
    // NO TOKEN LIMITS - NO TRUNCATION
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
    
    console.log(`✅ Success with ${modelState[nimModel].name}`);
    
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
  console.log(`🚀 NIM Proxy - 10 Request Batches with 15min Cooldown`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🔢 Batch size: ${MAX_REQUESTS_PER_BATCH} requests per model`);
  console.log(`🔒 Cooldown: ${COOLDOWN_MINUTES} minutes after ${MAX_REQUESTS_PER_BATCH} requests`);
  console.log(`🔄 Pattern: Pro (10) → Flash (10) → Pro (10) → Flash (10)...`);
  console.log(`🚫 429 Policy: NO RETRIES - Immediate 15 minute cooldown`);
  console.log(`⚠️ 500/503/504 Policy: Up to ${MAX_RETRIES} retries with backoff`);
  console.log(`📝 Full context - NO truncation, NO token limits`);
  console.log(`💡 Health check: /health`);
});
