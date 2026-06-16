// server.js - OpenAI to NVIDIA NIM API Proxy (Guaranteed Size Limits)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== CONFIGURATION - ADJUST THESE BASED ON YOUR NEEDS =====
const MAX_TOKENS_PER_REQUEST = 1500; // Hard limit - never exceed this
const MAX_RESPONSE_TOKENS = 1024;    // Keep responses short
const RATE_LIMIT_DELAY = 15000;      // 15 seconds between requests
const MAX_RETRIES = 2;               // Minimal retries to avoid rate limit spiral

// ===== SIMPLE DELAY-BASED RATE LIMITING =====
let lastRequestTime = 0;

async function enforceRateLimit() {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  
  if (timeSinceLastRequest < RATE_LIMIT_DELAY) {
    const waitTime = RATE_LIMIT_DELAY - timeSinceLastRequest;
    console.log(`⏳ Rate limit cooldown: Waiting ${Math.round(waitTime/1000)}s...`);
    await sleep(waitTime);
  }
  
  lastRequestTime = Date.now();
}

// ===== CONTEXT MANAGEMENT =====
class ContextManager {
  constructor() {
    this.conversationSummaries = new Map();
  }

  // Ultra-aggressive compression that GUARANTEES size limits
  prepareContext(messages) {
    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // If conversation is small enough, return as-is
    const totalTokens = estimateTokens(JSON.stringify(messages));
    if (totalTokens <= MAX_TOKENS_PER_REQUEST) {
      return messages;
    }
    
    console.log(`🔄 Compressing ${totalTokens} tokens to fit ${MAX_TOKENS_PER_REQUEST} limit...`);
    
    // STRATEGY: Keep only essential context
    const preparedMessages = [];
    
    // 1. Keep only the FIRST system message (truncated if needed)
    if (systemMessages.length > 0) {
      let sysContent = systemMessages[0].content;
      if (estimateTokens(sysContent) > 300) {
        sysContent = sysContent.substring(0, 1200) + '...'; // ~300 tokens
      }
      preparedMessages.push({
        role: 'system',
        content: sysContent
      });
    }
    
    // 2. Keep the LAST user message and the message before it (for context)
    const lastMessages = [];
    
    // Find the last user message
    let lastUserIndex = -1;
    for (let i = conversationMessages.length - 1; i >= 0; i--) {
      if (conversationMessages[i].role === 'user') {
        lastUserIndex = i;
        break;
      }
    }
    
    if (lastUserIndex >= 0) {
      // Include the message just before the last user message (if it exists)
      if (lastUserIndex > 0) {
        const prevMsg = conversationMessages[lastUserIndex - 1];
        lastMessages.push({
          role: prevMsg.role,
          content: this.truncateContent(prevMsg.content, 200)
        });
      }
      
      // Include the last user message
      lastMessages.push({
        role: 'user',
        content: this.truncateContent(conversationMessages[lastUserIndex].content, 500)
      });
      
      // Include any assistant messages after the last user message
      for (let i = lastUserIndex + 1; i < conversationMessages.length; i++) {
        lastMessages.push({
          role: conversationMessages[i].role,
          content: this.truncateContent(conversationMessages[i].content, 300)
        });
      }
    } else {
      // No user message found, just take last 2 messages
      const lastTwo = conversationMessages.slice(-2);
      lastMessages.push(...lastTwo.map(msg => ({
        role: msg.role,
        content: this.truncateContent(msg.content, 300)
      })));
    }
    
    preparedMessages.push(...lastMessages);
    
    // 3. Create a brief summary of older context
    if (lastUserIndex > 1) {
      const olderMessages = conversationMessages.slice(0, lastUserIndex - 1);
      const summary = this.createQuickSummary(olderMessages);
      
      // Insert summary as context between system and last messages
      preparedMessages.splice(1, 0, {
        role: 'system',
        content: `[Previous context: ${summary}]`
      });
    }
    
    // FINAL SIZE CHECK
    const finalTokens = estimateTokens(JSON.stringify(preparedMessages));
    console.log(`✅ Compressed to ${finalTokens} tokens (${preparedMessages.length} messages)`);
    
    // If somehow still too large, emergency cut
    if (finalTokens > MAX_TOKENS_PER_REQUEST) {
      console.log('🚨 Emergency: Still too large, keeping only last message');
      return [preparedMessages[0], preparedMessages[preparedMessages.length - 1]];
    }
    
    return preparedMessages;
  }
  
  truncateContent(content, maxTokens) {
    if (!content) return '';
    const maxChars = maxTokens * 4; // Rough char estimate
    if (content.length <= maxChars) return content;
    return content.substring(0, maxChars) + '...';
  }
  
  createQuickSummary(messages) {
    // Simple extraction-based summary (no API call needed)
    const userMessages = messages
      .filter(m => m.role === 'user')
      .slice(-5) // Last 5 user messages
      .map(m => m.content.substring(0, 100)) // First 100 chars of each
      .join(' | ');
    
    const assistantMessages = messages
      .filter(m => m.role === 'assistant')
      .slice(-3) // Last 3 assistant messages
      .map(m => {
        // Extract first sentence or key info
        const firstSentence = m.content.split(/[.!?]/)[0];
        return firstSentence ? firstSentence.substring(0, 80) : m.content.substring(0, 80);
      })
      .join(' | ');
    
    return `Recent topics: ${userMessages}. Previous responses: ${assistantMessages}`.substring(0, 500);
  }
}

const contextManager = new ContextManager();

// ===== HELPER FUNCTIONS =====

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function estimateTokens(text) {
  if (!text) return 0;
  const str = typeof text === 'string' ? text : JSON.stringify(text);
  return Math.ceil(str.length / 4);
}

async function callWithRetry(fn, context = '') {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const isRetryable = error.response?.status === 429 || 
                          error.response?.status === 503 || 
                          error.response?.status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }
      
      // On 429, wait longer
      const waitTime = 30000 * (attempt + 1); // 30s, 60s
      console.log(`⚠️ ${context} Rate limited (attempt ${attempt + 1}/${MAX_RETRIES + 1}), waiting ${waitTime/1000}s...`);
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); 
app.use(express.urlencoded({ limit: '50mb', extended: true }));

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

function resolveModel(openaiModel) {
  if (MODEL_MAPPING[openaiModel]) {
    return MODEL_MAPPING[openaiModel];
  }
  
  const modelLower = openaiModel.toLowerCase();
  if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus')) {
    return 'meta/llama-3.1-405b-instruct';
  } else if (modelLower.includes('claude') || modelLower.includes('gemini')) {
    return 'meta/llama-3.1-70b-instruct';
  }
  return 'meta/llama-3.1-8b-instruct';
}

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    max_tokens: MAX_TOKENS_PER_REQUEST,
    rate_limit_delay: `${RATE_LIMIT_DELAY/1000}s`,
    context_strategy: 'sliding_window_with_summary'
  });
});

// Models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
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
    
    if (!messages || messages.length === 0) {
      return res.status(400).json({
        error: { message: 'No messages provided', type: 'invalid_request_error', code: 400 }
      });
    }
    
    // Resolve model
    const nimModel = resolveModel(model);
    
    // GUARANTEED: Prepare context that fits within limits
    const preparedMessages = contextManager.prepareContext(messages);
    const inputTokens = estimateTokens(JSON.stringify(preparedMessages));
    
    console.log(`📊 Request: ${inputTokens} tokens, model: ${nimModel}`);
    
    // Prepare the request
    const nimRequest = {
      model: nimModel,
      messages: preparedMessages,
      temperature: temperature || 0.6,
      max_tokens: Math.min(max_tokens || MAX_RESPONSE_TOKENS, MAX_RESPONSE_TOKENS),
      stream: stream || false
    };
    
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    // Enforce rate limit
    await enforceRateLimit();
    
    // Make the request
    const response = await callWithRetry(
      () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 60000
      }),
      nimModel
    );
    
    console.log('✅ Request successful');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.on('data', (chunk) => {
        res.write(chunk);
      });
      
      response.data.on('end', () => res.end());
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
        model: model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: content
          },
          finish_reason: response.data.choices[0]?.finish_reason || 'stop'
        }],
        usage: response.data.usage || {
          prompt_tokens: inputTokens,
          completion_tokens: estimateTokens(content),
          total_tokens: inputTokens + estimateTokens(content)
        }
      });
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    
    const status = error.response?.status || 500;
    const retryAfter = error.response?.headers?.['retry-after'];
    
    // On 429, force a longer cooldown
    if (status === 429) {
      lastRequestTime = Date.now() + 60000; // Force 60s cooldown
    }
    
    res.status(status).json({
      error: {
        message: status === 429 
          ? 'Rate limit exceeded. Please wait before sending another message.'
          : error.message || 'Internal server error',
        type: status === 429 ? 'rate_limit_error' : 'server_error',
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
  console.log(`🚀 Proxy running on port ${PORT}`);
  console.log(`📦 Max input tokens: ${MAX_TOKENS_PER_REQUEST}`);
  console.log(`📝 Max response tokens: ${MAX_RESPONSE_TOKENS}`);
  console.log(`⏱️ Rate limit: 1 request per ${RATE_LIMIT_DELAY/1000}s`);
  console.log(`💡 Context: Sliding window with local summarization`);
});
