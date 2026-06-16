// server.js - OpenAI to NVIDIA NIM API Proxy (Full Context Preservation)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const pLimit = require('p-limit');

const app = express();
const PORT = process.env.PORT || 3000;

// Rate limiting setup
const limiter = pLimit(2);

// Retry configuration
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

// ===== CONTEXT MANAGEMENT CONFIGURATION =====
const MAX_TOKENS_PER_REQUEST = 4000;
const SUMMARY_TRIGGER_TOKENS = 3000; // Start summarizing when we hit this threshold
const SUMMARY_RESERVE_TOKENS = 1000; // Reserve tokens for the summary itself

// ===== CONTEXT SUMMARIZATION SYSTEM =====
class ConversationSummarizer {
  constructor() {
    this.summaries = new Map(); // Store summaries per conversation
  }

  async summarizeConversation(messages, nimModel) {
    // Extract key information from the conversation
    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // Create a summary prompt that captures all important context
    const summaryPrompt = {
      role: 'system',
      content: `You are a conversation summarizer. Create a detailed summary of the following conversation that captures:
1. ALL character personalities, traits, and backgrounds mentioned
2. ALL important events, plot points, and story developments
3. ALL key relationships between characters
4. ALL important decisions or choices made
5. The current situation and ongoing context
6. Any rules, settings, or world-building elements established

Be extremely detailed and comprehensive. This summary will replace the full conversation history to save context space while maintaining continuity.`
    };

    const conversationText = conversationMessages
      .map(m => `${m.role}: ${m.content}`)
      .join('\n\n');

    const summaryRequest = {
      model: nimModel,
      messages: [
        summaryPrompt,
        { role: 'user', content: `Please summarize this conversation while preserving ALL important context:\n\n${conversationText}` }
      ],
      temperature: 0.3,
      max_tokens: 2000
    };

    try {
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
        content: `[PREVIOUS CONVERSATION SUMMARY - Maintain continuity with this context]\n\n${summary}\n\n[END SUMMARY - Continue the conversation naturally based on this context]`
      };

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

    console.log(`🔄 Conversation too large (${estimatedTokens} tokens). Compressing...`);

    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // Keep the last few exchanges intact (most recent context)
    const RECENT_EXCHANGES = 3; // Keep last 3 complete exchanges
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
    
    // Summarize older messages
    if (olderMessages.length > 0) {
      console.log(`📝 Summarizing ${olderMessages.length} older messages...`);
      const summaryMessage = await this.summarizeConversation(
        [...systemMessages, ...olderMessages], 
        nimModel
      );
      
      if (summaryMessage) {
        // Combine: system messages + summary + recent messages
        const compressedMessages = [
          ...systemMessages,
          summaryMessage,
          ...recentMessages
        ];
        
        const compressedTokens = estimateTokens(JSON.stringify(compressedMessages));
        console.log(`✅ Compressed from ${estimatedTokens} to ${compressedTokens} tokens`);
        
        return compressedMessages;
      }
    }
    
    // If summarization fails, fall back to smart truncation
    console.log('⚠️ Falling back to smart truncation...');
    return this.smartTruncate(messages);
  }

  smartTruncate(messages) {
    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    const systemTokens = systemMessages.reduce((sum, msg) => sum + estimateTokens(msg.content), 0);
    let availableTokens = MAX_TOKENS_PER_REQUEST - systemTokens - 500;
    
    const selectedMessages = [];
    let tokensUsed = 0;
    
    // Work backwards to include most recent messages
    for (let i = conversationMessages.length - 1; i >= 0; i--) {
      const msgTokens = estimateTokens(conversationMessages[i].content);
      if (tokensUsed + msgTokens <= availableTokens) {
        selectedMessages.unshift(conversationMessages[i]);
        tokensUsed += msgTokens;
      } else {
        break;
      }
    }
    
    return [...systemMessages, ...selectedMessages];
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
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    bucket_tokens: requestBucket.tokens,
    bucket_capacity: requestBucket.capacity,
    max_tokens_per_request: MAX_TOKENS_PER_REQUEST,
    context_preservation: 'summarization'
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
    
    // MAIN FIX: Use summarization to preserve full context
    let processedMessages = messages;
    if (estimatedTokens > SUMMARY_TRIGGER_TOKENS) {
      console.log(`🔄 Conversation getting long (${estimatedTokens} tokens). Applying context preservation...`);
      processedMessages = await summarizer.compressConversation(messages, nimModel);
    }
    
    // Prepare the request
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
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
                
                // Filter out conversation completion attempts
                if (content && isFirstContent) {
                  isFirstContent = false;
                  if (content.startsWith('Human:') || content.startsWith('User:') || 
                      content.match(/^(Assistant|AI|Bot):/)) {
                    console.warn('⚠️ Filtered conversation completion prefix');
                    return;
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
      
      // Clean up any conversation completion artifacts
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
        console.log(`📊 Total tokens used: ${totalTokens}`);
      }
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('❌ Proxy error:', error.message);
    
    if (error.response) {
      console.error(`   Status: ${error.response.status}`);
      console.error(`   Data:`, JSON.stringify(error.response.data, null, 2));
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
  console.log(`🧠 Context preservation: Intelligent summarization`);
  console.log(`📝 Summarization triggers at: ${SUMMARY_TRIGGER_TOKENS} tokens`);
  console.log(`💾 Recent exchanges preserved: Last 3 exchanges always intact`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
  console.log(`🔍 Health check: http://localhost:${PORT}/health`);
});
