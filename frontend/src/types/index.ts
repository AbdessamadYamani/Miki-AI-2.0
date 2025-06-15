export interface Message {
  role: 'user' | 'assistant';
  content: string;
  references?: Array<{
    type: 'youtube';
    title: string;
    url: string;
    text: string;
  }>;
}

export interface Task {
  id: string;
  name: string;
  preview: string;
  full_plan: string;
}

export interface TaskSession {
  task_id: string;
  task_name: string;
  start_time: string;
  end_time: string | null;
  status: 'active' | 'paused' | 'completed' | 'failed';
  conversation_history: Message[];
  agent_thoughts: Array<{
    timestamp: string;
    content: string;
    type: string;
  }>;
  execution_log: string;
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
  taskHistory: TaskSession[];
  currentTaskId: string | null;
  currentTaskName: string;
  totalTokens: {
    prompt_tokens: number;
    candidates_tokens: number;
    total_tokens: number;
  };
}

export interface ApiResponse {
  success: boolean;
  message: string;
  tasks?: Task[];
  task?: Task;
} 