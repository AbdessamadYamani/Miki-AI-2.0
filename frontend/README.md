# AI Assistant Frontend

This is the frontend application for the AI Assistant, built with React, TypeScript, and Tailwind CSS. It provides a modern, responsive interface for interacting with the AI assistant and managing tasks.

## Features

- Real-time chat interface with the AI assistant
- Task management system
- Dark mode support
- Responsive design
- Markdown support in messages
- Code syntax highlighting

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

## Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Development

To start the development server:

```bash
npm start
```

The application will be available at `http://localhost:3000`.

## Building for Production

To create a production build:

```bash
npm run build
```

The build files will be created in the `build` directory.

## Project Structure

- `src/components/` - React components
- `src/services/` - API services
- `src/types/` - TypeScript type definitions
- `src/App.tsx` - Main application component
- `src/index.tsx` - Application entry point

## Technologies Used

- React
- TypeScript
- Tailwind CSS
- Material-UI Icons
- Axios
- React Markdown
- React Syntax Highlighter

## API Integration

The frontend communicates with the backend API running at `http://localhost:5001`. Make sure the backend server is running before starting the frontend application.
