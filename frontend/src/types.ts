export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  image?: {
    url: string;
    alt?: string;
  };
  references?: Array<{
    type: string;
    title: string;
    url: string;
  }>;
  image_data?: string;
  thoughts?: Array<{
    timestamp: string;
    content: string;
    type: string;
  }>;
}

export interface Task {
  id: string;
  name: string;
  preview: string;
  full_plan: string;
}

export interface ApiResponse {
  status: 'success' | 'error';
  message: string;
  [key: string]: any;
}

export interface UIState {
  userInputInteractive: boolean;
  sendInteractive: boolean;
  stopInteractive: boolean;
  stopVisible: boolean;
  continueInteractive: boolean;
  continueVisible: boolean;
  statusMessage: string;
  conversationHistory: Message[];
  executionLog: string;
  taskHistory: any[];
  currentTaskId: string | null;
  currentTaskName: string;
  totalTokens: {
    prompt_tokens: number;
    candidates_tokens: number;
    total_tokens: number;
  };
  isThinking: boolean;
  showThoughts: boolean;
  latestReasoning: string | null;
  agentThoughts: Array<{
    timestamp: string;
    content: string;
    type: string;
  }>;
  isSidebarOpen: boolean;
  isCodeViewOpen: boolean;
  isCommandPaletteOpen: boolean;
  showSettings: boolean;
} 