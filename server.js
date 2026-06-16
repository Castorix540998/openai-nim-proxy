// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const pLimit = require('p-limit');

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting setup - reduced to 1 concurrent request to minimize 429 errors
const limiter = pLimit(1); // Process one request at a time

// Retry configuration - more aggressive retry with longer waits
const RETRY_CONFIG = {
  maxRetries: 3,        // Reduced to 3 to avoid wasting time on huge requests
  initialDelay: 5000,   // Wait 5 seconds before first retry
  maxDelay: 30000,      // Wait up to 30 seconds
  backoffFactor: 2      // Double the wait time each retry
};

// ===== TOKEN-BASED THROTTLING CONFIGURATION =====
// This helps avoid invisible resource limits on NVIDIA's free tier
const TOKEN_CONFIG = {
  maxTokensPerSecond: 50,     // Reduced to 50 tokens/sec for more conservative throttling
  tokensUsedInLastSecond: 0,
  lastTokenResetTime: Date.now()
};

// ===== HELPER FUNCTIONS =====

// Sleep function
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// CORRECTED: Token-based throttling to avoid invisible resource limits
async function throttleTokens(tokensToUse) {
  const now = Date.now();
  const timeSinceReset = now - TOKEN_CONFIG.lastTokenResetTime;
  
  // Reset counter every second
  if (timeSinceReset >= 1000) {
    TOKEN_CONFIG.tokensUsedInLastSecond = 0;
    TOKEN_CONFIG.lastTokenResetTime = now;
    // Reset the timeSinceReset to 0 after reset
    timeSinceReset = 0;
  }
  
  // Check if we would exceed the limit
  if (TOKEN_CONFIG.tokensUsedInLastSecond + tokensToUse > TOKEN_CONFIG.maxTokensPerSecond) {
    // Calculate wait time - ensure it's always positive
    let waitTime = 1000 - timeSinceReset;
    if (waitTime < 100) waitTime = 100; // Minimum 100ms wait
    waitTime = Math.min(waitTime, 5000); // Maximum 5 seconds wait
    
    console.log(`⏳ Token throttling: ${tokensToUse} tokens would exceed ${TOKEN_CONFIG.maxTokensPerSecond}/s. Waiting ${waitTime}ms...`);
    await sleep(waitTime);
    // Reset after waiting
    TOKEN_CONFIG.tokensUsedInLastSecond = 0;
    TOKEN_CONFIG.lastTokenResetTime = Date.now();
  }
  
  TOKEN_CONFIG.tokensUsedInLastSecond += tokensToUse;
}

// Helper: Roughly estimate token count (approximation for throttling)
function estimateTokens(text) {
  if (!text) return 0;
  const length = typeof text === 'string' ? text.length : JSON.stringify(text).length;
  return Math.ceil(length / 4) + 10;
}

// Helper: Estimate tokens in messages
function estimateMessagesTokens(messages) {
  let total = 0;
  if (!messages || !Array.isArray(messages)) return 50;
  for (const msg of messages) {
    total += estimateTokens(msg.content || '');
    total += estimateTokens(msg.role || '');
    if (msg.name) total += estimateTokens(msg.name);
  }
  return total + 20;
}

// Request throttling - enforce minimum time between requests
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 5000; // Increased to 5 seconds between requests

// Helper: throttle requests to avoid rate limits
async function throttleRequest() {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
    const waitTime = MIN_REQUEST_INTERVAL - timeSinceLastRequest;
    console.log(`⏳ Request throttling: Waiting ${waitTime}ms before next request...`);
    await sleep(waitTime);
  }
  lastRequestTime = Date.now();
}

// Helper: call with retry for 429 errors
async function callWithRetry(fn, context = '') {
  let lastError;
  let delay = RETRY_CONFIG.initialDelay;
  
  for (let attempt = 1; attempt <= RETRY_CONFIG.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      // Only retry on 429, 503, 504
      const isRetryable = error.response?.status === 429 || 
                          error.response?.status === 503 || 
                          error.response?.status === 504 ||
                          error.code === 'ECONNRESET';
      
      if (!isRetryable || attempt === RETRY_CONFIG.maxRetries) {
        throw error;
      }
      
      // Check for Retry-After header
      let retryAfter = error.response?.headers?.['retry-after'];
      let waitTime = retryAfter ? parseInt(retryAfter) * 1000 : delay;
      waitTime = Math.min(waitTime, RETRY_CONFIG.maxDelay);
      
      console.log(`⚠️ ${context} Rate limited (attempt ${attempt}/${RETRY_CONFIG.maxRetries}), waiting ${waitTime}ms...`);
      await sleep(waitTime);
      delay = Math.min(delay * RETRY_CONFIG.backoffFactor, RETRY_CONFIG.maxDelay);
    }
  }
  throw lastError;
}

// Model cache to avoid repeated lookups
const modelCache = new Map();

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = false; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4': 'minimaxai/minimax-m3',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'z-ai/glm-5.1',
  'claude-3-sonnet': 'mistralai/mistral-medium-3.5-128b',
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b' 
};

// Helper function to resolve model without making test requests
async function resolveModel(openaiModel) {
  // Check cache first
  if (modelCache.has(openaiModel)) {
    return modelCache.get(openaiModel);
  }
  
  // Check mapping
  if (MODEL_MAPPING[openaiModel]) {
    modelCache.set(openaiModel, MODEL_MAPPING[openaiModel]);
    return MODEL_MAPPING[openaiModel];
  }
  
  // Use fallback based on model name patterns (without making API calls)
  const modelLower = openaiModel.toLowerCase();
  let fallbackModel;
  
  if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
    fallbackModel = 'meta/llama-3.1-405b-instruct';
  } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
    fallbackModel = 'meta/llama-3.1-70b-instruct';
  } else {
    fallbackModel = 'meta/llama-3.1-8b-instruct';
  }
  
  console.warn(`⚠️ Unknown model "${openaiModel}", using fallback: ${fallbackModel}`);
  modelCache.set(openaiModel, fallbackModel);
  return fallbackModel;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Get client IP for logging (optional)
    const clientIp = req.ip || req.connection.remoteAddress;
    
    // Smart model selection with fallback (NO test request)
    const nimModel = await resolveModel(model);
    
    // Check for huge request - warn if too large
    const estimatedInputTokens = estimateMessagesTokens(messages);
    console.log(`📊 Estimated input tokens: ${estimatedInputTokens}`);
    
    if (estimatedInputTokens > 2000) {
      console.warn(`⚠️ Large request detected: ${estimatedInputTokens} tokens. This may trigger rate limits.`);
      // For very large requests, add extra delay
      if (estimatedInputTokens > 5000) {
        const extraDelay = Math.min(10000, estimatedInputTokens * 2);
        console.log(`⏳ Large request: Adding ${extraDelay}ms extra delay...`);
        await sleep(extraDelay);
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };
    
    // Add thinking mode if enabled
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    // Throttle requests to avoid rate limits
    await throttleRequest();
    
    // Estimate token usage and throttle based on tokens (with reduced buffer for large requests)
    let tokenBuffer = 100;
    if (estimatedInputTokens > 5000) tokenBuffer = 50; // Smaller buffer for huge requests
    if (estimatedInputTokens > 10000) tokenBuffer = 20;
    
    await throttleTokens(estimatedInputTokens + tokenBuffer);
    
    // Make request to NVIDIA NIM API with concurrency limiting and retry logic
    const response = await limiter(async () => {
      return await callWithRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 120000 // 2 minute timeout
        }),
        `Model: ${nimModel}, IP: ${clientIp}`
      );
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      // Log token usage for monitoring
      if (response.data.usage) {
        const totalTokens = response.data.usage.total_tokens || 0;
        console.log(`📊 Total tokens used: ${totalTokens} (Prompt: ${response.data.usage.prompt_tokens || 0}, Completion: ${response.data.usage.completion_tokens || 0})`);
      }
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    // Better error response with Retry-After hint for 429 errors
    const status = error.response?.status || 500;
    const retryAfter = error.response?.headers?.['retry-after'];
    
    if (status === 429) {
      res.setHeader('Retry-After', retryAfter || '30');
    }
    
    res.status(status).json({
      error: {
        message: status === 429 
          ? 'Rate limit exceeded. Your request is too large or too frequent. Please wait and try again.'
          : error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 'invalid_request_error',
        code: status,
        retry_after: retryAfter ? parseInt(retryAfter) : 30
      }
    });
  }
});

// Catch-all for unsupported endpoints
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
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Rate limiting: 1 concurrent request, ${MIN_REQUEST_INTERVAL}ms between requests`);
  console.log(`Retry config: ${RETRY_CONFIG.maxRetries} retries, starting at ${RETRY_CONFIG.initialDelay}ms`);
  console.log(`Token throttling: ${TOKEN_CONFIG.maxTokensPerSecond} tokens/second`);
});
