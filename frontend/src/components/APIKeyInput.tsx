import React, { useState } from 'react';
import { TextField, Button, Box, Typography, Alert, CircularProgress } from '@mui/material';

interface ApiKeyInputProps {
  onApiKeyValid: () => void;
}

const ApiKeyInput: React.FC<ApiKeyInputProps> = ({ onApiKeyValid }) => {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const validateAndUpdateApiKey = async () => {
    setLoading(true);
    setError('');
    setSuccess(false);

    try {
      const response = await fetch('/update_api_key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ api_key: apiKey }),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess(true);
        // Wait a moment to show the success message before calling onApiKeyValid
        setTimeout(() => {
          onApiKeyValid();
        }, 1500);
      } else {
        setError(data.message || 'Invalid API key');
      }
    } catch (err) {
      setError('Failed to validate API key. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
        p: 3,
      }}
    >
      <Box
        sx={{
          maxWidth: 400,
          width: '100%',
          p: 4,
          borderRadius: 2,
          bgcolor: 'background.paper',
          boxShadow: 3,
        }}
      >
        <Typography variant="h5" component="h1" gutterBottom align="center">
          Welcome to Gemini Assistant
        </Typography>
        <Typography variant="body1" gutterBottom align="center" sx={{ mb: 3 }}>
          Please enter your Gemini API key to continue
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }}>
            API key validated and updated successfully!
          </Alert>
        )}

        <TextField
          fullWidth
          label="API Key"
          variant="outlined"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          disabled={loading}
          sx={{ mb: 2 }}
          type="password"
          helperText="Your API key will be validated and saved securely"
        />

        <Button
          fullWidth
          variant="contained"
          onClick={validateAndUpdateApiKey}
          disabled={loading || !apiKey}
          sx={{ height: 48 }}
        >
          {loading ? (
            <>
              <CircularProgress size={24} color="inherit" sx={{ mr: 1 }} />
              Validating...
            </>
          ) : (
            'Validate & Save'
          )}
        </Button>
      </Box>
    </Box>
  );
};

export default ApiKeyInput; 