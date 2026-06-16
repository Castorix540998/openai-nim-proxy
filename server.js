// server.js - OpenAI to NVIDIA NIM API Proxy (Aggressive Rate Limiting)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const pLimit = require('p-limit');

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting setup - SINGLE concurrent request to be safe
const limiter = pLimit(1);

// Retry configuration with MUCH longer waits
const RETRY_CONFIG = {
  maxRetries: 3,        // Reduced retries to avoid wasting time
  initialDelay: 10000,  // Start with 10 seconds
  maxDelay: 60000,      // Up to 60 seconds
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
    
    console.log(`⏳ Rate limit cooldown: Waiting ${Math.round(waitTime/1000)}s before next request...`);
    await sleep(Math.min(waitTime, 30000));
    this.refill();
    this.tokens -= tokens;
    return true;
  }
}

// VERY conservative rate limiting: 5 requests per minute
const requestBucket = new TokenBucket(5, 1, 12000); // 1 token every 12 seconds = 5/min

// ===== AGGRESSIVE CONTEXT MANAGEMENT =====
const MAX_TOKENS_PER_REQUEST = 2000; // MUCH more aggressive limit
const SUMMARY_TRIGGER_TOKENS = 1500; // Start summarizing earlier
const SUMMARY_MAX_TOKENS = 800; // Keep summaries focused

// ===== CONTEXT SUMMARIZATION SYSTEM =====
class ConversationSummarizer {
  constructor() {
    this.lastSummary = null;
    this.summaryCache = new Map();
  }

  async summarizeConversation(messages, nimModel) {
    // Check if we already have a recent summary
    const conversationKey = JSON.stringify(messages.slice(-5)); // Use last 5 messages as key
    if (this.summaryCache.has(conversationKey)) {
      console.log('📝 Using cached summary...');
      return this.summaryCache.get(conversationKey);
    }

    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // Create an ultra-focused summary prompt
    const summaryPrompt = {
      role: 'system',
      content: `Create a VERY concise summary (maximum 200 words) of this conversation. Include ONLY:
1. Current situation and immediate context
2. Key character details (names, personalities)
3. Last major event or decision
4. Active plot threads

Be extremely brief. This is for context preservation only.`
    };

    // Only summarize the older portion, not recent messages
    const olderMessages = conversationMessages.slice(0, -6); // Exclude last 3 exchanges
    if (olderMessages.length === 0) return null;

    const conversationText = olderMessages
      .slice(-20) // Only look at last 20 older messages
      .map(m => `${m.role}: ${m.content.substring(0, 200)}`) // Truncate each message
      .join('\n\n');

    const summaryRequest = {
      model: nimModel,
      messages: [
        summaryPrompt,
        { role: 'user', content: `Summarize this:\n\n${conversationText}` }
      ],
      temperature: 0.3,
      max_tokens: SUMMARY_MAX_TOKENS
    };

    try {
      // Wait for rate limit cooldown before summarization request
      await requestBucket.consume(1);
      
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, summaryRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        timeout: 60000
      });

      const summary = response.data.choices[0]?.message?.content || '';
      
      // Create a system message that includes the summary
      const contextMessage = {
        role: 'system',
        content: `[CONTEXT: ${summary.substring(0, 300)}]` // Even shorter context
      };

      // Cache the summary
      this.summaryCache.set(conversationKey, contextMessage);
      
      return contextMessage;
    } catch (error) {
      console.error('❌ Summarization failed:', error.message);
      return null;
    }
  }

  async compressConversation(messages, nimModel) {
    if (!messages || messages.length === 0) return messages;
    
    const estimatedTokens = estimateTokens(JSON.stringify(messages));
    
    // If under the limit, no compression needed
    if (estimatedTokens <= MAX_TOKENS_PER_REQUEST) {
      return messages;
    }

    console.log(`🔄 Conversation too large (${estimatedTokens} tokens). Aggressive compression...`);

    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // AGGRESSIVE: Keep only last 2 exchanges intact
    const RECENT_EXCHANGES = 2;
    let recentStartIndex = conversationMessages.length;
    let exchangeCount = 0;
    
    for (let i = conversationMessages.length - 1; i >= 0; i--) {
      if (conversationMessages[i].role === 'user') {
        exchangeCount++;
        if (exchangeCount >= RECENT_EXCHANGES) {
          recentStartIndex = i;
          break;
        }
      }
    }
    
    const olderMessages = conversationMessages.slice(0, recentStartIndex);
    const recentMessages = conversationMessages.slice(recentStartIndex);
    
    // Ultra-aggressive: If still too large, truncate recent messages
    let processedRecent = recentMessages;
    const recentTokens = estimateTokens(JSON.stringify([...systemMessages, ...recentMessages]));
    if (recentTokens > MAX_TOKENS_PER_REQUEST - 200) {
      // Truncate content of recent messages
      processedRecent = recentMessages.map(msg => ({
        ...msg,
        content: msg.content.substring(0, 300) + (msg.content.length > 300 ? '...' : '')
      }));
      console.log('⚠️ Truncated recent message contents to fit limits');
    }
    
    // Try to summarize older messages
    if (olderMessages.length > 0) {
      console.log(`📝 Summarizing ${olderMessages.length} older messages...`);
      const summaryMessage = await this.summarizeConversation(
        [...systemMessages, ...olderMessages], 
        nimModel
      );
      
      if (summaryMessage) {
        // Combine: system messages + summary + recent messages
        const compressedMessages = [
          ...systemMessages.slice(0, 1), // Keep only first system message
          summaryMessage,
          ...processedRecent
        ];
        
        const compressedTokens = estimateTokens(JSON.stringify(compressedMessages));
        console.log(`✅ Compressed from ${estimatedTokens} to ${compressedTokens} tokens`);
        
        // FINAL CHECK: If still too large, emergency truncation
        if (compressedTokens > MAX_TOKENS_PER_REQUEST) {
          console.log('🚨 Emergency truncation needed!');
          return [
            ...systemMessages.slice(0, 1),
            summaryMessage,
            processedRecent[processedRecent.length - 1] // Just the very last message
          ];
        }
        
        return compressedMessages;
      }
    }
    
    // Fallback: Ultra-aggressive truncation
    console.log('⚠️ Using emergency truncation...');
    return [
      ...systemMessages.slice(0, 1),
      ...processedRecent.slice(-4) // Just last 2 exchanges
    ];
  }
}

const summarizer = new ConversationSummarizer();

// ===== HELPER FUNCTIONS =====

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function estimateTokens(text) {
  if (!text) return 0;
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  return Math.ceil(str.length / 4);
}

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
      
      // FIXED: Better Retry-After handling
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ ${context} Got Retry-After header: ${retryAfter}s`);
      } else {
        // Exponential backoff with jitter, starting from 10s
        waitTime = RETRY_CONFIG.initialDelay * Math.pow(RETRY_CONFIG.backoffFactor, attempt);
        if (RETRY_CONFIG.jitter) {
          waitTime = waitTime * (0.5 + Math.random() * 0.5);
        }
        waitTime = Math.min(waitTime, RETRY_CONFIG.maxDelay);
      }
      
      console.log(`⚠️ ${context} Rate limited (attempt ${attempt + 1}/${RETRY_CONFIG.maxRetries + 1}), waiting ${Math.round(waitTime/1000)}s...`);
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
    bucket_tokens: requestBucket.tokens,
    bucket_capacity: requestBucket.capacity,
    max_tokens_per_request: MAX_TOKENS_PER_REQUEST,
    rate_limit: '5 requests/minute'
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
    
    // Smart model selection
    const nimModel = await resolveModel(model);
    
    // Estimate total tokens
    const estimatedTokens = estimateTokens(JSON.stringify(messages));
    console.log(`📊 Estimated input tokens: ${estimatedTokens}`);
    
    // AGGRESSIVE: Always compress if over limit
    let processedMessages = messages;
    if (estimatedTokens > MAX_TOKENS_PER_REQUEST) {
      console.log(`🔄 Compressing conversation (${estimatedTokens} tokens)...`);
      processedMessages = await summarizer.compressConversation(messages, nimModel);
      
      const compressedTokens = estimateTokens(JSON.stringify(processedMessages));
      console.log(`📦 Final request size: ${compressedTokens} tokens`);
      
      // If STILL too large after compression, reject with helpful error
      if (compressedTokens > MAX_TOKENS_PER_REQUEST * 1.2) {
        return res.status(413).json({
          error: {
            message: 'Conversation too large even after compression. Please start a new conversation or reduce context.',
            type: 'context_too_large',
            code: 413
          }
        });
      }
    }
    
    // Prepare the request with REDUCED max_tokens
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: Math.min(max_tokens || 2048, 2048), // MAX 2048 tokens for response
      stream: stream || false
    };
    
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    // Use token bucket for rate limiting (5 req/min)
    console.log('⏳ Waiting for rate limit cooldown...');
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
    
    // Reset bucket on successful request
    console.log('✅ Request successful');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.pipe(res);
      
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Stream error' });
        }
        res.end();
      });
    } else {
      let responseContent = response.data.choices[0]?.message?.content || '';
      
      // Clean up any artifacts
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
      
      if (response.data.usage) {
        console.log(`📊 Tokens: ${response.data.usage.total_tokens} total`);
      }
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('❌ Proxy error:', error.message);
    
    if (error.response) {
      console.error(`   Status: ${error.response.status}`);
      if (error.response.status === 429) {
        console.error('   ⚠️ RATE LIMITED - Request too large or too frequent');
        console.error('   Solution: Reduce conversation size or wait longer between requests');
      }
    }
    
    const status = error.response?.status || 500;
    const retryAfter = error.response?.headers?.['retry-after'];
    
    if (status === 429) {
      // Aggressive cooldown after 429
      requestBucket.tokens = 0;
      res.setHeader('Retry-After', '60'); // Force 60 second cooldown
    }
    
    res.status(status).json({
      error: {
        message: status === 429 
          ? 'Rate limit exceeded. Your request is too large. Try reducing conversation length or wait 60 seconds.'
          : error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 'invalid_request_error',
        code: status,
        retry_after: retryAfter ? parseInt(retryAfter) : 60
      }
    });
  }
});

// Catch-all
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
  console.log(`🐌 Rate limiting: 5 requests per minute (1 per 12 seconds)`);
  console.log(`📦 Max tokens per request: ${MAX_TOKENS_PER_REQUEST}`);
  console.log(`✂️ Aggressive compression: Enabled`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
});
