// server.js - OpenAI to NVIDIA NIM API Proxy (Improved Rate Limiting & Chunking)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const pLimit = require('p-limit');

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting setup - DYNAMIC concurrent requests
const limiter = pLimit(2); // Allow 2 concurrent requests initially

// Retry configuration with exponential backoff
const RETRY_CONFIG = {
  maxRetries: 5,        // More retries with better backoff
  initialDelay: 1000,   // Start with 1 second
  maxDelay: 60000,      // Up to 60 seconds
  backoffFactor: 2,     // Double each time
  jitter: true          // Add randomness to prevent thundering herd
};

// ===== TOKEN BUCKET FOR RATE LIMITING =====
class TokenBucket {
  constructor(capacity, refillRate, refillInterval = 1000) {
    this.capacity = capacity;
    this.tokens = capacity;
    this.refillRate = refillRate;
    this.refillInterval = refillInterval;
    this.lastRefill = Date.now();
  }

  refill() {
    const now = Date.now();
    const elapsed = now - this.lastRefill;
    const refillAmount = Math.floor(elapsed / this.refillInterval) * this.refillRate;
    this.tokens = Math.min(this.capacity, this.tokens + refillAmount);
    this.lastRefill = now;
  }

  async consume(tokens = 1) {
    this.refill();
    
    if (this.tokens >= tokens) {
      this.tokens -= tokens;
      return true;
    }
    
    // Calculate wait time until tokens are available
    const tokensNeeded = tokens - this.tokens;
    const waitTime = Math.ceil((tokensNeeded / this.refillRate) * this.refillInterval);
    
    console.log(`⏳ Token bucket: Need ${tokensNeeded} more tokens. Waiting ${waitTime}ms...`);
    await sleep(Math.min(waitTime, 10000)); // Max 10 second wait
    this.refill();
    this.tokens -= tokens;
    return true;
  }
}

// Create token bucket: 30 tokens max, refill 1 token per second (30 requests per minute)
const requestBucket = new TokenBucket(30, 1, 1000);

// ===== REQUEST CHUNKING CONFIGURATION =====
const MAX_TOKENS_PER_REQUEST = 4000; // Maximum tokens per request to avoid 429 errors

// ===== HELPER FUNCTIONS =====

// Sleep function
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Helper: Better token estimation
function estimateTokens(text) {
  if (!text) return 0;
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  // Better estimation: ~4 characters per token for English text
  return Math.ceil(str.length / 4);
}

// Helper: Split long conversation into chunks
function chunkMessages(messages, maxTokens = MAX_TOKENS_PER_REQUEST) {
  if (!messages || messages.length === 0) return [messages];
  
  const chunks = [];
  let currentChunk = [];
  let currentTokens = 0;
  
  // Always keep system message in every chunk
  const systemMessages = messages.filter(m => m.role === 'system');
  const nonSystemMessages = messages.filter(m => m.role !== 'system');
  
  // Calculate system message tokens
  const systemTokens = systemMessages.reduce((sum, msg) => 
    sum + estimateTokens(msg.content), 0);
  
  // Start first chunk with system messages
  currentChunk = [...systemMessages];
  currentTokens = systemTokens;
  
  for (let i = 0; i < nonSystemMessages.length; i++) {
    const msg = nonSystemMessages[i];
    const msgTokens = estimateTokens(msg.content) + estimateTokens(msg.role);
    
    // If adding this message would exceed the limit
    if (currentTokens + msgTokens > maxTokens && currentChunk.length > systemMessages.length) {
      // Save current chunk
      chunks.push([...currentChunk]);
      
      // Start new chunk with system messages + overlap for context
      const overlapStart = Math.max(0, currentChunk.length - systemMessages.length - 3);
      currentChunk = [
        ...systemMessages,
        ...currentChunk.slice(systemMessages.length + overlapStart)
      ];
      currentTokens = systemTokens + currentChunk
        .slice(systemMessages.length)
        .reduce((sum, m) => sum + estimateTokens(m.content), 0);
    }
    
    currentChunk.push(msg);
    currentTokens += msgTokens;
  }
  
  // Add final chunk if not empty
  if (currentChunk.length > systemMessages.length) {
    chunks.push(currentChunk);
  }
  
  return chunks.length > 0 ? chunks : [messages];
}

// Helper: Combine chunked responses
function combineChunkedResponses(responses) {
  if (responses.length === 1) return responses[0];
  
  // Combine all response content
  const combinedContent = responses
    .map(r => r.choices[0]?.message?.content || '')
    .join('\n\n---\n\n');
  
  // Sum up token usage
  const totalUsage = responses.reduce((sum, r) => ({
    prompt_tokens: (sum.prompt_tokens || 0) + (r.usage?.prompt_tokens || 0),
    completion_tokens: (sum.completion_tokens || 0) + (r.usage?.completion_tokens || 0),
    total_tokens: (sum.total_tokens || 0) + (r.usage?.total_tokens || 0)
  }), {});
  
  return {
    ...responses[0],
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: combinedContent
      },
      finish_reason: responses[responses.length - 1].choices[0]?.finish_reason || 'stop'
    }],
    usage: totalUsage
  };
}

// Helper: Call with smart retry logic
async function callWithRetry(fn, context = '') {
  let lastError;
  
  for (let attempt = 0; attempt <= RETRY_CONFIG.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      // Check if error is retryable
      const isRetryable = error.response?.status === 429 || 
                          error.response?.status === 503 || 
                          error.response?.status === 504 ||
                          error.code === 'ECONNRESET' ||
                          error.code === 'ETIMEDOUT';
      
      if (!isRetryable || attempt === RETRY_CONFIG.maxRetries) {
        throw error;
      }
      
      // Get retry delay from header or calculate with jitter
      let retryAfter = error.response?.headers?.['retry-after'];
      let waitTime;
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
      } else {
        // Exponential backoff with jitter
        waitTime = RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffFactor, attempt);
        if (RETRY_CONFIG.jitter) {
          waitTime = waitTime * (0.5 + Math.random() * 0.5); // 50-100% of calculated time
        }
        waitTime = Math.min(waitTime, RETRY_CONFIG.maxDelay);
      }
      
      console.log(`⚠️ ${context} Rate limited (attempt ${attempt + 1}/${RETRY_CONFIG.maxRetries + 1}), waiting ${Math.round(waitTime/1000)}s...`);
      console.log(`   Error details: ${error.response?.status} - ${error.response?.data?.error?.message || error.message}`);
      await sleep(waitTime);
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
    thinking_mode: ENABLE_THINKING_MODE,
    bucket_tokens: requestBucket.tokens,
    bucket_capacity: requestBucket.capacity,
    max_tokens_per_request: MAX_TOKENS_PER_REQUEST
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
    const clientIp = req.ip || req.connection.remoteAddress;
    
    // Smart model selection
    const nimModel = await resolveModel(model);
    
    // Estimate total tokens
    const estimatedTokens = estimateTokens(JSON.stringify(messages));
    console.log(`📊 Estimated total tokens: ${estimatedTokens}`);
    
    // If request is too large and not streaming, chunk it
    if (estimatedTokens > MAX_TOKENS_PER_REQUEST && !stream) {
      console.log(`🔄 Large request detected (${estimatedTokens} tokens). Splitting into chunks...`);
      
      const chunks = chunkMessages(messages);
      console.log(`📦 Split into ${chunks.length} chunks`);
      
      const chunkResponses = [];
      
      // Process each chunk with delay between them
      for (let i = 0; i < chunks.length; i++) {
        console.log(`🔄 Processing chunk ${i + 1}/${chunks.length} (${estimateTokens(JSON.stringify(chunks[i]))} tokens)`);
        
        const nimRequest = {
          model: nimModel,
          messages: chunks[i],
          temperature: temperature || 0.6,
          max_tokens: max_tokens || 9024,
          stream: false
        };
        
        if (ENABLE_THINKING_MODE) {
          nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
        }
        
        // Add delay between chunks to avoid rate limiting
        if (i > 0) {
          const chunkDelay = 2000 + (i * 1000); // Progressive delay
          console.log(`⏳ Waiting ${chunkDelay}ms before next chunk...`);
          await sleep(chunkDelay);
        }
        
        // Use token bucket for rate limiting
        await requestBucket.consume(1);
        
        const response = await limiter(async () => {
          return await callWithRetry(
            () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
              headers: {
                'Authorization': `Bearer ${NIM_API_KEY}`,
                'Content-Type': 'application/json'
              },
              timeout: 120000
            }),
            `Chunk ${i + 1}/${chunks.length} - Model: ${nimModel}`
          );
        });
        
        chunkResponses.push(response.data);
        console.log(`✅ Chunk ${i + 1}/${chunks.length} completed`);
      }
      
      // Combine responses
      const combinedResponse = combineChunkedResponses(chunkResponses);
      
      // Transform to OpenAI format
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: combinedResponse.choices.map(choice => {
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
        usage: combinedResponse.usage
      };
      
      console.log(`✅ Chunked request completed: ${combinedResponse.usage.total_tokens} total tokens`);
      return res.json(openaiResponse);
    }
    
    // For streaming or smaller requests, process normally
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };
    
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    // Use token bucket for rate limiting
    await requestBucket.consume(1);
    
    const response = await limiter(async () => {
      return await callWithRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 120000
        }),
        `Model: ${nimModel}`
      );
    });
    
    if (stream) {
      // Handle streaming response
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
        if (!res.headersSent) {
          res.status(500).json({ error: 'Stream error' });
        }
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
    console.error('❌ Proxy error:', error.message);
    
    // Enhanced error logging for debugging
    if (error.response) {
      console.error(`   Status: ${error.response.status}`);
      console.error(`   Headers:`, JSON.stringify(error.response.headers, null, 2));
      console.error(`   Data:`, JSON.stringify(error.response.data, null, 2));
    } else if (error.request) {
      console.error(`   No response received:`, error.code);
    }
    
    // Better error response with Retry-After hint for 429 errors
    const status = error.response?.status || 500;
    const retryAfter = error.response?.headers?.['retry-after'];
    
    if (status === 429) {
      // Empty token bucket after 429 to prevent cascade
      requestBucket.tokens = 0;
      res.setHeader('Retry-After', retryAfter || '30');
    }
    
    res.status(status).json({
      error: {
        message: error.response?.data?.error?.message || error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 
              status === 401 ? 'authentication_error' : 'invalid_request_error',
        code: status,
        retry_after: retryAfter ? parseInt(retryAfter) : null
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
  console.log(`🚀 OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`📊 Rate limiting: Token bucket (30 req/min, refill 1/sec)`);
  console.log(`🔄 Concurrent requests: 2 max`);
  console.log(`🔄 Retry config: ${RETRY_CONFIG.maxRetries + 1} attempts with exponential backoff`);
  console.log(`📦 Request chunking: Enabled for requests > ${MAX_TOKENS_PER_REQUEST} tokens`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
  console.log(`🔍 Health check: http://localhost:${PORT}/health`);
  console.log(`💭 Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🧠 Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
