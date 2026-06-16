// server.js - Smart Context Proxy with Triton Queue Management
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA TRITON QUEUE CONFIGURATION =====
// This tells NIM to accept more requests in queue instead of returning 429
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '100';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '8';

// ===== SMART CONTEXT CONFIGURATION =====
const MAX_TOKENS = 2500;           // Target token limit
const MAX_RESPONSE_TOKENS = 2048;
const MIN_DELAY = 8000;           // 8 seconds between requests
const MAX_RETRIES = 4;
const RETRY_BASE_DELAY = 5000;    // 5 second base for retry backoff

const IMPORTANCE_KEYWORDS = [      // Words that signal important context
  'name', 'remember', 'important', 'secret', 'promise',
  'backstory', 'character', 'personality', 'always', 'never',
  'love', 'hate', 'family', 'power', 'ability', 'rule',
  'remember that', 'don\'t forget', 'key', 'crucial', 'essential'
];

// ===== SMART CONTEXT MANAGER =====
class SmartContextManager {
  
  // Score messages by importance
  scoreImportance(message) {
    let score = 0;
    const content = (message.content || '').toLowerCase();
    
    // Recent messages are more important
    score += 10;
    
    // System messages are very important
    if (message.role === 'system') score += 50;
    
    // Messages with key details
    IMPORTANCE_KEYWORDS.forEach(keyword => {
      if (content.includes(keyword)) score += 15;
    });
    
    // Longer messages might contain more context
    if (content.length > 200) score += 10;
    if (content.length > 500) score += 5;
    
    // Messages with quotes or specific details
    if (content.includes('"') || content.includes("'")) score += 5;
    
    // Messages that seem to establish rules or facts
    if (content.match(/(is|are|was|were|will be|always|never) (a|the|an)/)) score += 10;
    
    return score;
  }
  
  prepareContext(messages) {
    const totalTokens = estimateTokens(JSON.stringify(messages));
    
    // If under limit, don't touch anything
    if (totalTokens <= MAX_TOKENS) {
      return messages;
    }
    
    console.log(`🧠 Smart compression: ${totalTokens} → ${MAX_TOKENS} tokens`);
    
    // Separate system messages (keep all)
    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // Score and sort conversation messages by importance
    const scoredMessages = conversationMessages.map((msg, index) => ({
      ...msg,
      _score: this.scoreImportance(msg),
      _index: index // Keep original order reference
    }));
    
    // Always keep the last 3 messages (immediate context)
    const lastThree = scoredMessages.slice(-3);
    const olderMessages = scoredMessages.slice(0, -3);
    
    // Sort older messages by importance score
    const importantOlder = olderMessages
      .sort((a, b) => b._score - a._score)
      .filter(msg => msg._score > 20); // Only keep moderately important ones
    
    // Combine: system + important older + recent
    let selectedMessages = [
      ...systemMessages,
      ...importantOlder.sort((a, b) => a._index - b._index), // Restore chronological order
      ...lastThree
    ];
    
    // Remove scoring metadata
    selectedMessages = selectedMessages.map(({ _score, _index, ...msg }) => msg);
    
    // If still too large, progressively trim older messages
    while (estimateTokens(JSON.stringify(selectedMessages)) > MAX_TOKENS) {
      const nonSystemNonRecent = selectedMessages.filter(
        m => m.role !== 'system' && !lastThree.includes(m)
      );
      
      if (nonSystemNonRecent.length === 0) {
        // Emergency: truncate content of recent messages
        selectedMessages = selectedMessages.map(msg => ({
          ...msg,
          content: msg.content.substring(0, Math.floor(msg.content.length * 0.7))
        }));
      } else {
        // Remove least important older message
        const leastImportant = nonSystemNonRecent.reduce((min, msg) => 
          (this.scoreImportance(msg) < this.scoreImportance(min)) ? msg : min
        );
        selectedMessages = selectedMessages.filter(m => m !== leastImportant);
      }
    }
    
    const finalTokens = estimateTokens(JSON.stringify(selectedMessages));
    console.log(`✅ Preserved ${selectedMessages.length} messages (${finalTokens} tokens)`);
    
    return selectedMessages;
  }
}

const contextManager = new SmartContextManager();

// ===== REQUEST QUEUE WITH TRITON SUPPORT =====
class RequestQueue {
  constructor() {
    this.queue = [];
    this.processing = false;
    this.lastRequestTime = 0;
  }

  async add(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject });
      this.process();
    });
  }

  async process() {
    if (this.processing || this.queue.length === 0) return;
    
    this.processing = true;
    
    while (this.queue.length > 0) {
      const { fn, resolve, reject } = this.queue.shift();
      
      try {
        // Wait rate limit delay between requests
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        
        if (timeSinceLastRequest < MIN_DELAY) {
          const waitTime = MIN_DELAY - timeSinceLastRequest;
          console.log(`⏳ Queue: Waiting ${Math.round(waitTime/1000)}s...`);
          await sleep(waitTime);
        }
        
        this.lastRequestTime = Date.now();
        const result = await fn();
        resolve(result);
      } catch (error) {
        reject(error);
      }
    }
    
    this.processing = false;
  }
}

const requestQueue = new RequestQueue();

// ===== HELPERS =====
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(JSON.stringify(text).length / 4);
}

async function callWithRetry(fn, context = '') {
  let lastError;
  
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const status = error.response?.status;
      const isRetryable = status === 429 || status === 503 || status === 504;
      
      if (!isRetryable || attempt === MAX_RETRIES) {
        throw error;
      }
      
      // Check for Retry-After header (Triton provides this)
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ Triton Retry-After: ${retryAfter}s`);
      } else {
        // Exponential backoff with jitter
        waitTime = RETRY_BASE_DELAY * Math.pow(2, attempt);
        waitTime = waitTime * (0.8 + Math.random() * 0.4);
      }
      
      console.log(`⚠️ ${context} attempt ${attempt + 1}/${MAX_RETRIES + 1} (${status}). Waiting ${Math.round(waitTime/1000)}s...`);
      await sleep(waitTime);
    }
  }
  throw lastError;
}

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-v4-flash',
  'gpt-4': 'minimaxai/minimax-m3',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'z-ai/glm-5.1',
  'claude-3-sonnet': 'mistralai/mistral-medium-3.5-128b',
  'gemini-pro': 'nvidia/nemotron-3-ultra-550b-a55b'
};

function resolveModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  const lower = model.toLowerCase();
  if (lower.includes('gpt-4') || lower.includes('opus')) return 'meta/llama-3.1-405b-instruct';
  if (lower.includes('claude') || lower.includes('gemini')) return 'meta/llama-3.1-70b-instruct';
  return 'meta/llama-3.1-8b-instruct';
}

// Health check with Triton info
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    strategy: 'smart_importance_scoring',
    max_tokens: MAX_TOKENS,
    min_delay: `${MIN_DELAY/1000}s`,
    triton_queue_size: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
    triton_batch_size: process.env.NIM_TRITON_MAX_BATCH_SIZE,
    queue_length: requestQueue.queue.length,
    max_retries: MAX_RETRIES
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
      return res.status(400).json({ error: { message: 'No messages', code: 400 } });
    }
    
    const nimModel = resolveModel(model);
    const preparedMessages = contextManager.prepareContext(messages);
    
    console.log(`📤 Sending ${estimateTokens(JSON.stringify(preparedMessages))} tokens to ${nimModel}`);
    
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
    
    // Use queue + retry with Triton support
    const response = await requestQueue.add(() =>
      callWithRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 120000 // 2 minute timeout for queued requests
        }),
        nimModel
      )
    );
    
    console.log('✅ Success');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      response.data.on('data', (chunk) => res.write(chunk));
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });
    } else {
      let content = response.data.choices[0]?.message?.content || '';
      
      // Clean up any conversation completion artifacts
      content = content.replace(/^(?:Human|User|Assistant|AI|Bot):\s*/gm, '');
      
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
        usage: response.data.usage || {
          prompt_tokens: estimateTokens(JSON.stringify(preparedMessages)),
          completion_tokens: estimateTokens(content),
          total_tokens: estimateTokens(JSON.stringify(preparedMessages)) + estimateTokens(content)
        }
      });
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    
    const status = error.response?.status || 500;
    const retryAfter = error.response?.headers?.['retry-after'];
    
    if (status === 429) {
      console.log('💡 Triton queue full. Try increasing NIM_TRITON_MAX_QUEUE_SIZE');
    }
    
    res.status(status).json({
      error: {
        message: status === 429 
          ? 'Server busy. Request queued. Try increasing NIM_TRITON_MAX_QUEUE_SIZE.'
          : error.response?.data?.error?.message || error.message,
        type: status === 429 ? 'rate_limit_error' : 'server_error',
        code: status,
        retry_after: retryAfter ? parseInt(retryAfter) : undefined
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: 'Not found', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`🚀 Smart Context Proxy with Triton Support`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🧠 Strategy: Importance-based context preservation`);
  console.log(`📦 Max tokens: ${MAX_TOKENS}`);
  console.log(`📝 Max response: ${MAX_RESPONSE_TOKENS} tokens`);
  console.log(`⏱️ Rate: 1 request per ${MIN_DELAY/1000}s`);
  console.log(`📋 Triton queue: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
  console.log(`📊 Triton batch: ${process.env.NIM_TRITON_MAX_BATCH_SIZE}`);
  console.log(`🔄 Max retries: ${MAX_RETRIES}`);
  console.log(`🔑 API Base: ${NIM_API_BASE}`);
});
