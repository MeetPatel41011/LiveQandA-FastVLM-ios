import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import CopyWebpackPlugin from 'copy-webpack-plugin';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer, webpack }) => {
    // 1. Enable async WebAssembly
    config.experiments = { ...config.experiments, asyncWebAssembly: true };

    // 2. Resolve fallbacks to suppress Node.js built-in warnings in browser
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };

    config.plugins.push(
      new webpack.IgnorePlugin({
        resourceRegExp: /canvas/,
        contextRegExp: /jsdom$/,
      }),
      new webpack.IgnorePlugin({
        resourceRegExp: /^(fsevents|pnpapi)$/,
      })
    );

    // 3. Copy ONNX Runtime WASM files to the public static directory
    if (!isServer) {
      config.plugins.push(
        new CopyWebpackPlugin({
          patterns: [
            {
              from: resolve(__dirname, 'node_modules/onnxruntime-web/dist/*.wasm'),
              to: resolve(__dirname, '.next/static/wasm/[name][ext]'),
            },
            {
              from: resolve(__dirname, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm'),
              to: resolve(__dirname, '.next/static/wasm/ort-wasm-simd-threaded.jsep.wasm'),
            },
            {
              from: resolve(__dirname, 'node_modules/onnxruntime-web/dist/ort-wasm-simd.jsep.wasm'),
              to: resolve(__dirname, '.next/static/wasm/ort-wasm-simd.jsep.wasm'),
            },
            {
              from: resolve(__dirname, 'node_modules/onnxruntime-web/dist/ort-wasm.jsep.wasm'),
              to: resolve(__dirname, '.next/static/wasm/ort-wasm.jsep.wasm'),
            },
          ],
        })
      );
    }

    return config;
  },
};

export default nextConfig;