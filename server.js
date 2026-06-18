// server.js - NVIDIA NIM Proxy - Strict Alternation
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA CONFIGURATION =====
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '500';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '16';
process.env.NIM_MAX_CPU_LORAS = process.env.NIM_MAX_CPU_LORAS || '10';

// ===== STRICT ALTERNATION - Pro → Flash → Pro → Flash... =====
const MODEL_PRO = 'deepseek-ai/deepseek-v4-pro';
const MODEL_FLASH = 'deepseek-ai/deepseek-v4-flash';

// Track when each model was last used
let lastProTime = 0;
let lastFlashTime = 0;
const MIN_SAME_MODEL_DELAY = 61000; // 61 SECONDS between same model

// Track which model to use next (start with Pro)
let useProNext = true;

function getNextModel() {
  const now = Date.now();
  
  if (useProNext) {
    useProNext = false;
    return { model: MODEL_PRO, lastUsed: lastProTime };
  } else {
    useProNext = true;
    return { model: MODEL_FLASH, lastUsed: lastFlashTime };
  }
}

function updateModelTime(model) {
  const now = Date.now();
  if (model === MODEL_PRO) {
    lastProTime = now;
  } else {
    lastFlashTime = now;
  }
}

// ===== RATE LIMITING WITH 61-SECOND SAME-MODEL SPACING =====
let activeRequests = 0;
const requestQueue = [];
let processingQueue = false;

async function processQueue() {
  if (processingQueue || requestQueue.length === 0) return;
  
  processingQueue = true;
  
  while (requestQueue.length > 0) {
    const { resolve, reject } = requestQueue.shift();
    
    try {
      // Get the next model in alternation
      const { model, lastUsed } = getNextModel();
      const now = Date.now();
      const timeSinceLastUse = now - lastUsed;
      
      // Check if we need to wait for 61-second spacing
      if (timeSinceLastUse < MIN_SAME_MODEL_DELAY) {
        const waitTime = MIN_SAME_MODEL_DELAY - timeSinceLastUse;
        console.log(`⏳ ${model.split('/').pop()}: Waiting ${Math.round(waitTime/1000)}s (61s spacing between same model)...`);
        await sleep(waitTime);
      }
      
      // Update the last used time for this model
      updateModelTime(model);
      
      console.log(`📤 Using: ${model.split('/').pop()}`);
      resolve(model);
      
      // Small gap between queue items
      await sleep(500);
      
    } catch (error) {
      reject(error);
    }
  }
  
  processingQueue = false;
}

function waitForTurn() {
  return new Promise((resolve, reject) => {
    requestQueue.push({ resolve, reject });
    processQueue();
  });
}

// ===== DETAILED 429 LOGGING =====
let last429Error = null;

// ===== RETRY CONFIGURATION =====
const MAX_RETRIES = 2;
const RETRY_BASE_DELAY = 5000;

// ===== HELPERS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function callWithRetry(fn, model) {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      
      if (status === 429) {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`🚫 429 on ${model.split('/').pop()}`);
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
          model: model,
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
      
      console.log(`⚠️ Server error on ${model.split('/').pop()} - retry ${attempt + 1}/${MAX_RETRIES + 1}. Waiting ${Math.round(waitTime/1000)}s...`);
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

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': MODEL_FLASH,
  'gpt-4': MODEL_PRO,
  'gpt-4-turbo': MODEL_FLASH,
  'gpt-4o': MODEL_PRO,
  'claude-3-opus': MODEL_PRO,
  'claude-3-sonnet': MODEL_FLASH,
  'gemini-pro': MODEL_PRO
};

// Health check
app.get('/health', (req, res) => {
  const now = Date.now();
  const timeSincePro = now - lastProTime;
  const timeSinceFlash = now - lastFlashTime;
  
  res.json({
    status: 'ok',
    alternation: 'Pro → Flash → Pro → Flash...',
    min_same_model_spacing: `${MIN_SAME_MODEL_DELAY / 1000}s`,
    next_model: useProNext ? 'deepseek-v4-pro' : 'deepseek-v4-flash',
    last_pro: `${Math.round(timeSincePro / 1000)}s ago`,
    last_flash: `${Math.round(timeSinceFlash / 1000)}s ago`,
    pro_available_in: Math.max(0, Math.round((MIN_SAME_MODEL_DELAY - timeSincePro) / 1000)) + 's',
    flash_available_in: Math.max(0, Math.round((MIN_SAME_MODEL_DELAY - timeSinceFlash) / 1000)) + 's',
    queue_length: requestQueue.length,
    active_requests: activeRequests,
    last_429: last429Error
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
    
    // Wait for turn in the alternation queue
    console.log(`📥 Request queued (${requestQueue.length + 1} in queue)`);
    const nimModel = await waitForTurn();
    
    activeRequests++;
    console.log(`📤 ${messages.length} messages → ${nimModel.split('/').pop()}`);
    
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

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, code: 404 } });
});

app.listen(PORT, () => {
  console.log(`🚀 NIM Proxy - Strict Pro ↔ Flash Alternation`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🔄 Pattern: Pro → Flash → Pro → Flash → Pro...`);
  console.log(`⏱️ Same model spacing: 61 SECONDS minimum`);
  console.log(`📝 Full context - NO truncation, NO token limits`);
  console.log(`💡 Requests queue and wait for their turn`);
});
