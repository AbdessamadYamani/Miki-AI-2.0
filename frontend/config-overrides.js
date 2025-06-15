const webpack = require('webpack');

module.exports = function override(config) {
  // Add fallbacks for Node.js core modules
  config.resolve.fallback = {
    "http": require.resolve("stream-http"),
    "https": require.resolve("https-browserify"),
    "util": require.resolve("util/"),
    "zlib": require.resolve("browserify-zlib"),
    "stream": require.resolve("stream-browserify"),
    "url": require.resolve("url/"),
    "crypto": require.resolve("crypto-browserify"),
    "assert": require.resolve("assert/"),
    "buffer": require.resolve("buffer/"),
    "process": false,
  };

  // Add plugins
  config.plugins = [
    ...config.plugins,
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
    }),
    new webpack.DefinePlugin({
      'process.env': JSON.stringify(process.env),
    }),
  ];

  // Add process polyfill to module rules
  config.module.rules.push({
    test: /\.m?js/,
    resolve: {
      fullySpecified: false
    }
  });

  // Add resolve aliases
  config.resolve.alias = {
    ...config.resolve.alias,
    'process': 'process/browser',
  };

  // Add process polyfill to entry point
  if (Array.isArray(config.entry)) {
    config.entry.unshift('process/browser');
  } else if (typeof config.entry === 'object') {
    Object.keys(config.entry).forEach(key => {
      if (Array.isArray(config.entry[key])) {
        config.entry[key].unshift('process/browser');
      }
    });
  }

  return config;
}; 