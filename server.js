// server.js - Narrative-Aware Context Proxy with Triton Queue Management
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ===== NVIDIA TRITON QUEUE CONFIGURATION =====
process.env.NIM_TRITON_MAX_QUEUE_SIZE = process.env.NIM_TRITON_MAX_QUEUE_SIZE || '100';
process.env.NIM_TRITON_MAX_BATCH_SIZE = process.env.NIM_TRITON_MAX_BATCH_SIZE || '8';

// ===== CONTEXT CONFIGURATION =====
const MAX_TOKENS = 2500;
const MAX_RESPONSE_TOKENS = 2048;
const MIN_DELAY = 8000;
const MAX_RETRIES = 4;
const RETRY_BASE_DELAY = 5000;

// ===== NARRATIVE-AWARE CONTEXT MANAGER =====
class NarrativeContextManager {
  
  prepareContext(messages) {
    const totalTokens = estimateTokens(JSON.stringify(messages));
    
    // If under limit, don't touch anything
    if (totalTokens <= MAX_TOKENS) {
      return messages;
    }
    
    console.log(`📖 Preserving narrative flow: ${totalTokens} → ${MAX_TOKENS} tokens`);
    
    const systemMessages = messages.filter(m => m.role === 'system');
    const conversationMessages = messages.filter(m => m.role !== 'system');
    
    // CRITICAL: Keep messages in CHRONOLOGICAL ORDER
    // Never sort by importance - that breaks the story flow
    
    // Find complete exchanges (user-assistant pairs)
    const exchanges = [];
    let currentExchange = [];
    
    for (const msg of conversationMessages) {
      currentExchange.push(msg);
      if (msg.role === 'assistant') {
        exchanges.push([...currentExchange]);
        currentExchange = [];
      }
    }
    
    // If there's an incomplete exchange (last message is from user), add it
    if (currentExchange.length > 0) {
      exchanges.push([...currentExchange]);
    }
    
    // Strategy: Keep ALL system messages + last N complete exchanges
    const systemTokens = estimateTokens(JSON.stringify(systemMessages));
    let availableTokens = MAX_TOKENS - systemTokens - 100; // Reserve 100 for safety
    let selectedExchanges = [];
    let usedTokens = 0;
    
    // Always include the LAST exchange (current conversation)
    if (exchanges.length > 0) {
      const lastExchange = exchanges[exchanges.length - 1];
      const lastTokens = estimateTokens(JSON.stringify(lastExchange));
      selectedExchanges.unshift(lastExchange);
      usedTokens += lastTokens;
      availableTokens -= lastTokens;
    }
    
    // Work backwards, keeping complete exchanges until we run out of tokens
    for (let i = exchanges.length - 2; i >= 0; i--) {
      const exchange = exchanges[i];
      const exchangeTokens = estimateTokens(JSON.stringify(exchange));
      
      if (usedTokens + exchangeTokens <= availableTokens) {
        selectedExchanges.unshift(exchange); // Add to front to maintain order
        usedTokens += exchangeTokens;
      } else {
        // Try to include a condensed version of this exchange
        const condensedExchange = exchange.map(msg => ({
          role: msg.role,
          content: this.condenseMessage(msg)
        }));
        const condensedTokens = estimateTokens(JSON.stringify(condensedExchange));
        
        if (usedTokens + condensedTokens <= availableTokens) {
          selectedExchanges.unshift(condensedExchange);
          usedTokens += condensedTokens;
        } else {
          // Can't fit this exchange at all, stop here
          break;
        }
      }
    }
    
    // Flatten exchanges back into message array
    let selectedMessages = [
      ...systemMessages,
      ...selectedExchanges.flat()
    ];
    
    // Add a narrative continuity marker if we had to cut context
    if (exchanges.length > selectedExchanges.length) {
      const removedCount = exchanges.length - selectedExchanges.length;
      selectedMessages.splice(systemMessages.length, 0, {
        role: 'system',
        content: `[Continuing the ongoing story. Previous ${removedCount} exchanges happened earlier. Maintain story continuity naturally.]`
      });
    }
    
    // Add a final instruction to keep the model in "story mode"
    selectedMessages.push({
      role: 'system',
      content: '[Continue the story naturally from the last exchange. Do not summarize or describe characters unless asked. Stay in character and advance the plot.]'
    });
    
    const finalTokens = estimateTokens(JSON.stringify(selectedMessages));
    console.log(`✅ Kept ${selectedExchanges.length}/${exchanges.length} exchanges (${finalTokens} tokens)`);
    
    return selectedMessages;
  }
  
  condenseMessage(msg) {
    // Keep the essence of the message without full detail
    const content = msg.content || '';
    if (content.length <= 300) return content;
    
    // Keep first and last parts of long messages
    const first = content.substring(0, 150);
    const last = content.substring(content.length - 150);
    return `${first}... [continues] ...${last}`;
  }
}

const contextManager = new NarrativeContextManager();

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
      
      let waitTime;
      const retryAfter = error.response?.headers?.['retry-after'];
      
      if (retryAfter) {
        waitTime = parseInt(retryAfter) * 1000;
        console.log(`⚠️ Triton Retry-After: ${retryAfter}s`);
      } else {
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

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok',
    strategy: 'narrative_exchange_preservation',
    max_tokens: MAX_TOKENS,
    min_delay: `${MIN_DELAY/1000}s`,
    triton_queue_size: process.env.NIM_TRITON_MAX_QUEUE_SIZE,
    queue_length: requestQueue.queue.length
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
    
    const response = await requestQueue.add(() =>
      callWithRetry(
        () => axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 120000
        }),
        nimModel
      )
    );
    
    console.log('✅ Success');
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let isFirstChunk = true;
      
      response.data.on('data', (chunk) => {
        if (isFirstChunk) {
          // Check if model is trying to summarize instead of continue
          const chunkStr = chunk.toString();
          if (chunkStr.includes('character') && chunkStr.includes('description') ||
              chunkStr.includes('summary') || chunkStr.includes('recap')) {
            console.warn('⚠️ Detected summary mode, attempting to redirect...');
            // Don't filter, just warn - the narrative markers should help
          }
          isFirstChunk = false;
        }
        
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            res.write(line + '\n');
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });
    } else {
      let content = response.data.choices[0]?.message?.content || '';
      
      // Clean up any summary artifacts
      content = content.replace(/^(?:Here is a |Let me |I'll ).*?(?:summary|description|recap).*?:?\s*/i, '');
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
  console.log(`🚀 Narrative-Aware Proxy with Triton Support`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`📖 Strategy: Complete exchange preservation (chronological)`);
  console.log(`📦 Max tokens: ${MAX_TOKENS}`);
  console.log(`⏱️ Rate: 1 request per ${MIN_DELAY/1000}s`);
  console.log(`📋 Triton queue: ${process.env.NIM_TRITON_MAX_QUEUE_SIZE}`);
});
