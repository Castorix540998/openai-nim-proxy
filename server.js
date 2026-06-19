// server.js - NVIDIA NIM Proxy - 3 Model Rotation with Hourly Limits
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
const MODEL_GLM = 'z-ai/glm-5.1';

// ===== BATCH SYSTEM =====
const COOLDOWN_MINUTES = 60;
const COOLDOWN_MS = COOLDOWN_MINUTES * 60 * 1000;

// Track each model's usage and limits
const modelState = {
  [MODEL_PRO]: {
    requestsUsed: 0,
    cooldownUntil: 0,
    name: 'DeepSeek V4 Pro',
    maxRequests: 10,
    retriesAllowed: 1
  },
  [MODEL_FLASH]: {
    requestsUsed: 0,
    cooldownUntil: 0,
    name: 'DeepSeek V4 Flash',
    maxRequests: 10,
    retriesAllowed: 1
  },
  [MODEL_GLM]: {
    requestsUsed: 0,
    cooldownUntil: 0,
    name: 'GLM 5.1',
    maxRequests: 30,
    retriesAllowed: 2
  }
};

// ===== SEQUENCE: Pro → GLM → Flash → GLM → Pro → GLM → Flash → GLM... =====
const sequence = [
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  MODEL_PRO, MODEL_GLM,
  MODEL_FLASH, MODEL_GLM,
  // GLM solo until its 30 are used
  MODEL_GLM, MODEL_GLM, MODEL_GLM, MODEL_GLM, MODEL_GLM,
  MODEL_GLM, MODEL_GLM, MODEL_GLM, MODEL_GLM, MODEL_GLM,
  MODEL_GLM
];

let sequenceIndex = 0;

function getNextModelInSequence() {
  const now = Date.now();
  
  // Check if all models are available (cooldowns expired and counters reset)
  const allAvailable = Object.values(modelState).every(state => {
    if (now < state.cooldownUntil) return false;
    if (state.requestsUsed >= state.maxRequests && now >= state.cooldownUntil) {
      // Reset counter if cooldown has expired
      state.requestsUsed = 0;
    }
    return state.requestsUsed < state.maxRequests;
  });
  
  // If all are fresh, restart sequence from beginning
  if (allAvailable && sequenceIndex >= sequence.length) {
    console.log(`🔄 All models available! Restarting sequence from beginning.`);
    sequenceIndex = 0;
  }
  
  // Try to find the next available model in sequence
  for (let i = 0; i < sequence.length; i++) {
    const modelIndex = (sequenceIndex + i) % sequence.length;
    const model = sequence[modelIndex];
    const state = modelState[model];
    
    // Check if model is on cooldown
    if (now < state.cooldownUntil) {
      continue; // Skip, on cooldown
    }
    
    // Check if model has requests remaining
    if (state.requestsUsed >= state.maxRequests) {
      continue; // Skip, no requests left
    }
    
    // Found an available model
    sequenceIndex = (modelIndex + 1) % sequence.length; // Point to next for future
    return model;
  }
  
  // No model available
  return null;
}

function incrementRequestCount(model) {
  modelState[model].requestsUsed++;
  const state = modelState[model];
  const remaining = state.maxRequests - state.requestsUsed;
  
  if (state.requestsUsed >= state.maxRequests) {
    // Put model on cooldown
    state.cooldownUntil = Date.now() + COOLDOWN_MS;
    console.log(`🔒 ${state.name}: ${state.requestsUsed}/${state.maxRequests} requests used - COOLDOWN for ${COOLDOWN_MINUTES} minutes (until ${new Date(state.cooldownUntil).toLocaleTimeString()})`);
  } else {
    console.log(`📊 ${state.name}: ${state.requestsUsed}/${state.maxRequests} requests used (${remaining} remaining)`);
  }
}

function forceCooldown(model) {
  const state = modelState[model];
  state.cooldownUntil = Date.now() + COOLDOWN_MS;
  state.requestsUsed = state.maxRequests; // Mark as fully used
  console.log(`🔒 ${state.name} FORCED into ${COOLDOWN_MINUTES} minute cooldown (until ${new Date(state.cooldownUntil).toLocaleTimeString()})`);
}

// ===== HELPER FUNCTIONS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ===== RETRY CONFIGURATION =====
async function callWithRetry(fn, model) {
  const state = modelState[model];
  const maxRetries = state.retriesAllowed;
  let lastError;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      
      // ===== 429 HANDLING: NO RETRIES, FORCE COOLDOWN =====
      if (status === 429) {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🚫 429 RATE LIMIT on ${state.name}`);
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
        
        // IMMEDIATELY force cooldown
        forceCooldown(model);
        console.log(`🚫 NO RETRIES - Model locked out for ${COOLDOWN_MINUTES} minutes`);
        console.log(`${'='.repeat(60)}\n`);
        throw error;
      }
      
      // ===== 500, 503, 504 HANDLING =====
      const isRetryable = status === 500 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === maxRetries) {
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
      
      console.log(`⚠️ Server error (${status}) on ${state.name} - retry ${attempt + 1}/${maxRetries + 1}. Waiting ${Math.round(waitTime/1000)}s...`);
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
  const now = Date.now();
  
  const getModelStatus = (modelKey) => {
    const state = modelState[modelKey];
    const cooldownRemaining = Math.max(0, state.cooldownUntil - now);
    return {
      requests_used: `${state.requestsUsed}/${state.maxRequests}`,
      requests_remaining: state.maxRequests - state.requestsUsed,
      on_cooldown: now < state.cooldownUntil,
      cooldown_remaining: cooldownRemaining > 0 ? `${Math.ceil(cooldownRemaining / 1000)}s (${Math.ceil(cooldownRemaining / 60000)} minutes)` : 'none',
      available: now >= state.cooldownUntil && state.requestsUsed < state.maxRequests,
      retries_allowed: state.retriesAllowed
    };
  };
  
  res.json({
    status: 'ok',
    next_in_sequence: modelState[sequence[sequenceIndex]]?.name || 'end of sequence',
    sequence_position: `${sequenceIndex}/${sequence.length}`,
    deepseek_v4_pro: getModelStatus(MODEL_PRO),
    deepseek_v4_flash: getModelStatus(MODEL_FLASH),
    glm_5_1: getModelStatus(MODEL_GLM),
    system: {
      cooldown_minutes: COOLDOWN_MINUTES,
      sequence: 'Pro → GLM → Flash → GLM (repeating) → GLM solo (final 11)',
      retry_policy: {
        pro_flash: '1 retry on 500/503/504',
        glm: '2 retries on 500/503/504',
        on_429: 'NO RETRIES - Immediate 60 minute cooldown'
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
    
    // Get next model in sequence
    const nimModel = getNextModelInSequence();
    
    // If no model is available (all on cooldown or out of requests)
    if (!nimModel) {
      const now = Date.now();
      
      // Find the soonest available model
      let soonestModel = null;
      let soonestTime = Infinity;
      
      for (const [key, state] of Object.entries(modelState)) {
        const cooldownRemaining = Math.max(0, state.cooldownUntil - now);
        if (cooldownRemaining < soonestTime) {
          soonestTime = cooldownRemaining;
          soonestModel = state.name;
        }
      }
      
      const minutesRemaining = Math.ceil(soonestTime / 60000);
      const secondsRemaining = Math.ceil(soonestTime / 1000);
      
      console.log(`🚫 All models unavailable! Blocking request BEFORE reaching NVIDIA.`);
      
      return res.status(429).json({
        error: {
          message: `All models are currently on cooldown. ${soonestModel} will be available in ${minutesRemaining} minute(s) (${secondsRemaining} seconds).`,
          type: 'rate_limit_error',
          code: 429,
          retry_after: secondsRemaining,
          details: {
            deepseek_v4_pro: modelState[MODEL_PRO].requestsUsed >= modelState[MODEL_PRO].maxRequests ? 'cooldown' : 'available',
            deepseek_v4_flash: modelState[MODEL_FLASH].requestsUsed >= modelState[MODEL_FLASH].maxRequests ? 'cooldown' : 'available',
            glm_5_1: modelState[MODEL_GLM].requestsUsed >= modelState[MODEL_GLM].maxRequests ? 'cooldown' : 'available'
          }
        }
      });
    }
    
    const state = modelState[nimModel];
    
    console.log(`📤 ${messages.length} messages → ${state.name} (${state.requestsUsed + 1}/${state.maxRequests})`);
    
    // Increment request counter
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
    
    console.log(`✅ Success with ${state.name}`);
    
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
  console.log(`🚀 NIM Proxy - 3 Model Rotation (60min Cooldowns)`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🔢 Limits:`);
  console.log(`   • DeepSeek Pro:  ${modelState[MODEL_PRO].maxRequests} requests/hour, ${modelState[MODEL_PRO].retriesAllowed} retry`);
  console.log(`   • DeepSeek Flash: ${modelState[MODEL_FLASH].maxRequests} requests/hour, ${modelState[MODEL_FLASH].retriesAllowed} retry`);
  console.log(`   • GLM 5.1:       ${modelState[MODEL_GLM].maxRequests} requests/hour, ${modelState[MODEL_GLM].retriesAllowed} retries`);
  console.log(`🔒 Cooldown: ${COOLDOWN_MINUTES} minutes after limit reached`);
  console.log(`🔄 Sequence: Pro → GLM → Flash → GLM (repeating) → GLM solo`);
  console.log(`🛡️ Cooldown models blocked BEFORE reaching NVIDIA`);
  console.log(`🚫 429: NO RETRIES - Immediate cooldown`);
  console.log(`📝 Full context - NO truncation, NO token limits`);
  console.log(`💡 Health check: /health`);
});
