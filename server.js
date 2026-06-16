// server.js - OpenAI to NVIDIA NIM API Proxy (Fixed Context & Chunking)
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
  maxRetries: 5,
  initialDelay: 1000,
  maxDelay: 60000,
  backoffFactor: 2,
  jitter: true
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
    
    const tokensNeeded = tokens - this.tokens;
    const waitTime = Math.ceil((tokensNeeded / this.refillRate) * this.refillInterval);
    
    console.log(`⏳ Token bucket: Need ${tokensNeeded} more tokens. Waiting ${waitTime}ms...`);
    await sleep(Math.min(waitTime, 10000));
    this.refill();
    this.tokens -= tokens;
    return true;
  }
}

const requestBucket = new TokenBucket(30, 1, 1000);

// ===== IMPROVED REQUEST CHUNKING CONFIGURATION =====
const MAX_TOKENS_PER_REQUEST = 4000;

// ===== HELPER FUNCTIONS =====

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Better token estimation
function estimateTokens(text) {
  if (!text) return 0;
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  return Math.ceil(str.length / 4);
}

// IMPROVED: Smart conversation chunking that preserves context
function smartChunkConversation(messages, maxTokens = MAX_TOKENS_PER_REQUEST) {
  if (!messages || messages.length === 0) return null;
  
  // If messages fit within limit, no chunking needed
  const totalTokens = estimateTokens(JSON.stringify(messages));
  if (totalTokens <= maxTokens) return null;
  
  const systemMessages = messages.filter(m => m.role === 'system');
  const conversationMessages = messages.filter(m => m.role !== 'system');
  
  // Find the last user message
  const lastUserMessageIndex = [...conversationMessages].reverse().findIndex(m => m.role === 'user');
  if (lastUserMessageIndex === -1) return null; // No user message found
  
  const actualLastUserIndex = conversationMessages.length - 1 - lastUserMessageIndex;
  
  // Strategy: Keep all system messages + last N messages that fit within limit
  const systemTokens = systemMessages.reduce((sum, msg) => sum + estimateTokens(msg.content), 0);
  let availableTokens = maxTokens - systemTokens - 500; // Reserve 500 tokens for response
  
  // Start from the end and work backwards to include most recent context
  const selectedMessages = [];
  let tokensUsed = 0;
  
  // Always include the last user message and any assistant responses after it
  for (let i = actualLastUserIndex; i < conversationMessages.length; i++) {
    const msgTokens = estimateTokens(conversationMessages[i].content);
    if (tokensUsed + msgTokens <= availableTokens) {
      selectedMessages.push(conversationMessages[i]);
      tokensUsed += msgTokens;
    }
  }
  
  // Then add previous messages from the last user message backwards
  let includedCount = selectedMessages.length;
  for (let i = actualLastUserIndex - 1; i >= 0 && tokensUsed < availableTokens; i--) {
    const msgTokens = estimateTokens(conversationMessages[i].content);
    if (tokensUsed + msgTokens <= availableTokens) {
      selectedMessages.unshift(conversationMessages[i]);
      tokensUsed += msgTokens;
      includedCount++;
    } else {
      break;
    }
  }
  
  // If we couldn't even fit the last exchange, just keep the minimum
  if (selectedMessages.length < 2) {
    const lastExchange = conversationMessages.slice(Math.max(0, actualLastUserIndex - 1));
    return [...systemMessages, ...lastExchange];
  }
  
  console.log(`✂️ Smart truncation: Kept ${includedCount}/${conversationMessages.length} messages (${tokensUsed} tokens)`);
  return [...systemMessages, ...selectedMessages];
}

// Helper: Call with smart retry logic
async function callWithRetry(fn, context = '') {
  let lastError;
  
  for (let attempt = 0; attempt <= RETRY_CONFIG.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const isRetryable = error.response?.status === 429 || 
                          error.response?.status === 503 || 
                          error.response?.status === 504 ||
                          error.code === 'ECONNRESET' ||
                          error.code === 'ETIMEDOUT';
      
      if (!isRetryable || attempt === RETRY_CONFIG.maxRetries) {
        throw error;
      }
      
      let retryAfter = error.response?.headers?.['retry-after'];
      let waitTime;
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
      } else {
        waitTime = RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffFactor, attempt);
        if (RETRY_CONFIG.jitter) {
          waitTime = waitTime * (0.5 + Math.random() * 0.5);
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

// Model cache
const modelCache = new Map();

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

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

async function resolveModel(openaiModel) {
  if (modelCache.has(openaiModel)) {
    return modelCache.get(openaiModel);
  }
  
  if (MODEL_MAPPING[openaiModel]) {
    modelCache.set(openaiModel, MODEL_MAPPING[openaiModel]);
    return MODEL_MAPPING[openaiModel];
  }
  
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

// List models endpoint
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

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const clientIp = req.ip || req.connection.remoteAddress;
    
    // Smart model selection
    const nimModel = await resolveModel(model);
    
    // Estimate total tokens
    const estimatedTokens = estimateTokens(JSON.stringify(messages));
    console.log(`📊 Estimated total tokens: ${estimatedTokens}`);
    
    // FIXED: Use smart truncation instead of chunking to preserve context
    let processedMessages = messages;
    if (estimatedTokens > MAX_TOKENS_PER_REQUEST && !stream) {
      console.log(`🔄 Large request detected (${estimatedTokens} tokens). Smart truncation...`);
      const truncatedMessages = smartChunkConversation(messages);
      if (truncatedMessages) {
        processedMessages = truncatedMessages;
        console.log(`📦 Reduced to ${estimateTokens(JSON.stringify(processedMessages))} tokens`);
      }
    }
    
    // Prepare the request
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096, // Reduced from 9024 to be safer
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
      let isFirstContent = true;
      
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
                
                // FIX: Ensure we're not repeating user's messages in the response
                if (content && isFirstContent) {
                  isFirstContent = false;
                  // Check if the model is trying to continue the conversation instead of responding
                  if (content.startsWith('Human:') || content.startsWith('User:') || 
                      content.includes(messages[messages.length - 1]?.content?.substring(0, 20))) {
                    console.warn('⚠️ Detected model trying to complete user message, filtering...');
                    return; // Skip this chunk
                  }
                }
                
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
      // Transform NIM response to OpenAI format
      let responseContent = response.data.choices[0]?.message?.content || '';
      
      // FIX: Additional check to ensure model isn't completing the conversation
      const lastUserMessage = messages[messages.length - 1]?.content || '';
      if (responseContent.includes(lastUserMessage.substring(0, 50))) {
        console.warn('⚠️ Response contains user message, cleaning up...');
        // Try to extract just the assistant's part
        const assistantParts = responseContent.split(/\n(?=(?:Assistant|AI|Bot): )/);
        if (assistantParts.length > 1) {
          responseContent = assistantParts[assistantParts.length - 1];
        }
      }
      
      // Remove any "Human:" or "User:" prefixes the model might generate
      responseContent = responseContent.replace(/^(?:Human|User|Assistant|AI|Bot):\s*/gm, '');
      
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: responseContent
          },
          finish_reason: response.data.choices[0]?.finish_reason || 'stop'
        }],
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      // Handle reasoning content if enabled
      if (SHOW_REASONING && response.data.choices[0]?.message?.reasoning_content) {
        const reasoning = response.data.choices[0].message.reasoning_content;
        openaiResponse.choices[0].message.content = 
          '<think>\n' + reasoning + '\n</think>\n\n' + responseContent;
      }
      
      // Log token usage
      if (response.data.usage) {
        const totalTokens = response.data.usage.total_tokens || 0;
        console.log(`📊 Total tokens used: ${totalTokens} (Prompt: ${response.data.usage.prompt_tokens || 0}, Completion: ${response.data.usage.completion_tokens || 0})`);
      }
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('❌ Proxy error:', error.message);
    
    if (error.response) {
      console.error(`   Status: ${error.response.status}`);
      console.error(`   Headers:`, JSON.stringify(error.response.headers, null, 2));
      console.error(`   Data:`, JSON.stringify(error.response.data, null, 2));
    } else if (error.request) {
      console.error(`   No response received:`, error.code);
    }
    
    const status = error.response?.status || 500;
    const retryAfter = error.response?.headers?.['retry-after'];
    
    if (status === 429) {
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
  console.log(`📦 Smart truncation: Enabled for requests > ${MAX_TOKENS_PER_REQUEST} tokens`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
  console.log(`🔍 Health check: http://localhost:${PORT}/health`);
  console.log(`💭 Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🧠 Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🛡️ Anti-completion protection: Enabled`);
});
