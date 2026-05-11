import { pipeline, env } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web/webgpu';

// Configure ONNX Runtime WebAssembly paths mapped from webpack
ort.env.wasm.wasmPaths = '/_next/static/wasm/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

// Configure Transformers.js
env.allowRemoteModels = true;
env.allowLocalModels = false;

let inferencePipeline = null;
let currentResolve = null;

self.onmessage = async (event) => {
  const { type, payload } = event.data;

  if (type === 'loadModel') {
    try {
      console.log('Worker: Loading WebGPU Model...');
      // Load Xenova's LLaVA model configured for WebGPU
      inferencePipeline = await pipeline(
        payload.task || 'image-text-to-text',
        payload.modelName || 'Xenova/llava-onevision',
        {
          device: 'webgpu',
          dtype: 'fp16', 
        }
      );
      self.postMessage({ type: 'modelLoaded', success: true });
      console.log('Worker: Model loaded successfully (WebGPU).');
    } catch (error) {
      console.error('Worker: Failed to load model:', error);
      self.postMessage({ type: 'modelLoaded', success: false, error: error.message });
    }
  } 
  else if (type === 'runInference') {
    if (!inferencePipeline) {
      self.postMessage({ type: 'inferenceResult', success: false, error: 'Model not loaded.' });
      return;
    }
    try {
      console.log('Worker: Running inference...');
      
      const messages = [
        {
          "role": "user",
          "content": [
            { "type": "image" },
            { "type": "text", "text": "Analyze this image and decide if it needs 'web_search', 'calculator', or 'none'. Extract text word-for-word. Output JSON format: {\"extracted_question\": \"...\", \"tool_needed\": \"...\", \"tool_query\": \"...\", \"answer\": \"...\"}" }
          ]
        }
      ];
      
      const modelOutput = await inferencePipeline(messages, payload.imageBlob);
      const rawTextOutput = Array.isArray(modelOutput) && modelOutput.length > 0
        ? modelOutput[0].generated_text || JSON.stringify(modelOutput[0])
        : typeof modelOutput === 'string' ? modelOutput : JSON.stringify(modelOutput);

      const toolCall = parseToolCall(rawTextOutput);

      if (toolCall && toolCall.type === 'web_search') {
        console.log(`Worker: Intercepted tool call for web search: ${toolCall.query}`);
        // Dispatch to Main Thread to execute secure Serverless Route
        self.postMessage({ type: 'toolCall', payload: toolCall });

        // Wait for Main Thread to fetch Tavily and resolve promise
        const toolResult = await new Promise(resolve => { currentResolve = resolve; });

        console.log('Worker: Received Tavily data. Resuming generation...');
        
        const followUpMessages = [
           ...messages,
           { "role": "assistant", "content": rawTextOutput },
           { "role": "user", "content": `Search Results:\n${toolResult.results}\nBased on this, what is the final direct answer?` }
        ];
        
        const finalOutput = await inferencePipeline(followUpMessages);
        self.postMessage({ type: 'inferenceResult', success: true, result: finalOutput });

      } else {
        self.postMessage({ type: 'inferenceResult', success: true, result: rawTextOutput });
      }
      console.log('Worker: Inference complete.');
    } catch (error) {
      console.error('Worker: Failed to run inference:', error);
      self.postMessage({ type: 'inferenceResult', success: false, error: error.message });
    }
  } 
  else if (type === 'toolResult') {
    if (currentResolve) {
      currentResolve(payload);
      currentResolve = null;
    }
  }
};

function parseToolCall(outputText) {
  try {
    const jsonStart = outputText.lastIndexOf('{');
    if (jsonStart !== -1) {
      const jsonStr = outputText.substring(jsonStart);
      const parsed = JSON.parse(jsonStr.replace(/`/g, ''));
      if (parsed.tool_needed === 'web_search' && parsed.tool_query) {
        return { type: 'web_search', query: parsed.tool_query };
      }
    }
  } catch (e) {
    // Silent fail to regex
  }
  return null;
}