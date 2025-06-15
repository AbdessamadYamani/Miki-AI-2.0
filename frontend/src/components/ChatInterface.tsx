import React, { useState, useRef, useEffect, useCallback } from 'react';
import { UIState, Message } from '../types';
import ChatMessage from './ChatMessage';
import { sendMessage, startNewTask, stopTask, continueTask, getTaskHistory } from '../services/api';
import { 
  Send as SendIcon, 
  Stop as StopIcon, 
  PlayArrow as PlayIcon, 
  Add as AddIcon,
  AttachFile as AttachIcon,
  Code as CodeIcon,
  Settings as SettingsIcon,
  Search as SearchIcon,
  EmojiEmotions as EmojiIcon,
  ArrowDownward as ArrowDownIcon,
  Close as CloseIcon,
  Error as ErrorIcon,
  Chat as ChatIcon,
  List as ListIcon,
  Folder as FolderIcon,
  Assessment as AssessmentIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import MermaidSlidePanel from './MermaidSlidePanel';

interface ChatInterfaceProps {
  activeTab: 'chat' | 'tasks';
  onTabChange: (tab: 'chat' | 'tasks') => void;
}

interface TaskSession {
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

interface AgentThought {
  timestamp: string;
  content: string;
  type: string;
}

const API_BASE_URL = 'http://localhost:5001';

const ChatInterface: React.FC<ChatInterfaceProps> = ({ activeTab, onTabChange }) => {
  const [uiState, setUiState] = useState<UIState>({
    userInputInteractive: true,
    sendInteractive: true,
    stopInteractive: false,
    stopVisible: true,
    continueInteractive: false,
    continueVisible: false,
    statusMessage: 'Ready for new task',
    conversationHistory: [],
    executionLog: '',
    taskHistory: [],
    currentTaskId: null,
    currentTaskName: 'None',
    totalTokens: {
      prompt_tokens: 0,
      candidates_tokens: 0,
      total_tokens: 0
    },
    isThinking: false,
    showThoughts: false,
    latestReasoning: null,
    agentThoughts: [],
    isSidebarOpen: true,
    isCodeViewOpen: false,
    isCommandPaletteOpen: false,
    showSettings: false
  });

  const [inputMessage, setInputMessage] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isCodeViewOpen, setIsCodeViewOpen] = useState(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [commandInput, setCommandInput] = useState('');
  const [filePreview, setFilePreview] = useState<File | null>(null);
  const [isExecutingTask, setIsExecutingTask] = useState(false);
  const [taskSessions, setTaskSessions] = useState<TaskSession[]>([]);
  const [selectedThought, setSelectedThought] = useState<AgentThought[] | null>(null);
  const [isLoadingThoughts, setIsLoadingThoughts] = useState(false);
  const [isDiagramPanelOpen, setIsDiagramPanelOpen] = useState(false);
  const [allDiagrams, setAllDiagrams] = useState<{ code: string; index: number }[]>([]);
  const [currentTaskDiagrams, setCurrentTaskDiagrams] = useState<{ [taskId: string]: { code: string; index: number }[] }>({});
  const [showProcessingDetails, setShowProcessingDetails] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [apiKeyStatus, setApiKeyStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [apiKeyMessage, setApiKeyMessage] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Add new state for tracking last update
  const [lastUpdateTime, setLastUpdateTime] = useState<number>(Date.now());
  const [isPolling, setIsPolling] = useState<boolean>(false);

  // Task suggestions
  const taskSuggestions = [
    {
      id: 'suggest1',
      name: 'File Organization',
      description: 'Organize files in a specific directory',
      icon: <FolderIcon className="w-5 h-5" />,
      color: 'from-blue-500 to-blue-600',
      prompt: 'Please help me organize my files in the Documents folder by creating appropriate subfolders and moving files accordingly.'
    },
    {
      id: 'suggest2',
      name: 'Data Analysis',
      description: 'Analyze CSV data and generate reports',
      icon: <AssessmentIcon className="w-5 h-5" />,
      color: 'from-green-500 to-green-600',
      prompt: 'Can you analyze this CSV file and generate a summary report with key insights?'
    },
    {
      id: 'suggest3',
      name: 'System Check',
      description: 'Check system health and performance',
      icon: <SpeedIcon className="w-5 h-5" />,
      color: 'from-purple-500 to-purple-600',
      prompt: 'Please check my system performance and provide recommendations for optimization.'
    },
  ];

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
      if (window.innerWidth < 768) {
        setIsSidebarOpen(false);
        setIsCodeViewOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [uiState.conversationHistory, scrollToBottom]);

  useEffect(() => {
    const handleScroll = () => {
      if (!messagesContainerRef.current) return;
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      setShowScrollToBottom(scrollHeight - scrollTop - clientHeight > 100);
    };

    const container = messagesContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, []);

  // Add this effect to update task sessions when UI state changes
  useEffect(() => {
    if (uiState.taskHistory) {
      setTaskSessions(uiState.taskHistory);
    }
  }, [uiState.taskHistory]);

  // Add effect to preserve task state when switching tabs
  useEffect(() => {
    if (activeTab === 'chat' && uiState.currentTaskId) {
      const fetchTaskState = async () => {
        try {
          console.log('Fetching task state for ID:', uiState.currentTaskId);
          const response = await fetch(`${API_BASE_URL}/tasks/${uiState.currentTaskId}`);
          if (response.ok) {
            const taskState = await response.json();
            console.log('Received task state:', JSON.stringify({
              taskId: taskState.task_id,
              taskName: taskState.task_name,
              status: taskState.status,
              historyLength: taskState.conversation_history?.length,
              historyPreview: taskState.conversation_history?.map((msg: { role: 'user' | 'assistant'; content: string }) => ({
                role: msg.role,
                contentLength: msg.content.length,
                contentPreview: msg.content.substring(0, 100) + '...',
                hasMermaid: msg.content.includes('--- START MERMAID CODE ---') || msg.content.includes('```mermaid')
              }))
            }, null, 2));

            // Ensure we have the full conversation history
            if (taskState.conversation_history && Array.isArray(taskState.conversation_history)) {
              setUiState(prevState => ({
                ...prevState,
                conversationHistory: taskState.conversation_history,
                currentTaskId: taskState.task_id,
                currentTaskName: taskState.task_name,
                statusMessage: taskState.status === 'paused' ? 'Waiting for input' : 
                             taskState.status === 'active' ? 'Processing task' : 
                             taskState.status === 'completed' ? 'Task completed' : 
                             taskState.status === 'failed' ? 'Task failed' : 'Ready for new task'
              }));
            } else {
              console.error('Invalid conversation history format:', taskState.conversation_history);
            }
          }
        } catch (error) {
          console.error('Error fetching task state:', error);
        }
      };
      fetchTaskState();
    }
  }, [activeTab, uiState.currentTaskId]);

  // Add effect to log conversation history changes
  useEffect(() => {
    if (uiState.conversationHistory.length > 0) {
      console.log('Conversation history updated:', JSON.stringify({
        length: uiState.conversationHistory.length,
        messages: uiState.conversationHistory.map(msg => ({
          role: msg.role,
          contentLength: msg.content.length,
          contentPreview: msg.content.substring(0, 100) + '...',
          hasMermaidMarkers: msg.content.includes('--- START MERMAID CODE ---'),
          hasMarkdownMermaid: msg.content.includes('```mermaid'),
          fullContent: msg.content // Include full content for debugging
        }))
      }, null, 2));
    }
  }, [uiState.conversationHistory]);

  // Add initial load of task history
  useEffect(() => {
    const loadInitialTaskHistory = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/tasks/history`);
        if (response.ok) {
          const history = await response.json();
          setTaskSessions(history);
          // Also update UI state with the history
          setUiState(prevState => ({
            ...prevState,
            taskHistory: history
          }));
        }
      } catch (error) {
        console.error('Error loading initial task history:', error);
        setError('Failed to load task history');
      }
    };

    loadInitialTaskHistory();
  }, []); // Empty dependency array means this runs once on mount

  // Update handleNewTask to include task name
  const handleNewTask = async (taskName?: string): Promise<UIState> => {
    setError(null);
    try {
      const newState = await startNewTask(taskName);
      setUiState(newState);
      setFilePreview(null);
      // Clear diagrams for the new task
      setAllDiagrams([]);
      setIsDiagramPanelOpen(false);
      // Clear input if it was used as task name
      if (taskName) {
        setInputMessage('');
      }
      return newState;
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to start new task');
      console.error('Error starting new task:', error);
      throw error;
    }
  };

  // Update handleTaskHistoryClick to handle diagrams
  const handleTaskHistoryClick = async (taskId: string) => {
    console.log('Attempting to load and set task history for:', taskId);
    setError(null);
    setIsLoadingThoughts(true);
    try {
      const setCurrentResponse = await fetch(`${API_BASE_URL}/tasks/resume/${taskId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!setCurrentResponse.ok) {
        const errorBody = await setCurrentResponse.text();
        console.error(`Failed to resume task ${taskId}. Status: ${setCurrentResponse.status}. Body: ${errorBody}`);
        throw new Error(`Failed to resume task ${taskId}: ${setCurrentResponse.statusText}`);
      }

      const newState: UIState = await setCurrentResponse.json();
      console.log('Backend resumed task. Received new state:', newState);

      setUiState(newState);
      
      if (newState.taskHistory) {
        setTaskSessions(newState.taskHistory);
      }

      // Update diagrams for the current task
      if (currentTaskDiagrams[taskId]) {
        setAllDiagrams(currentTaskDiagrams[taskId]);
      } else {
        setAllDiagrams([]);
      }
      setIsDiagramPanelOpen(false);

      console.log(`Task with ID ${taskId} loaded and set as current.`);

    } catch (error) {
      console.error(`Error loading and setting task history for ${taskId}:`, error);
      setError(error instanceof Error ? error.message : 'Failed to load task history.');
    } finally {
      setIsLoadingThoughts(false);
    }
  };

  // Modify the polling effect
  useEffect(() => {
    let pollInterval: NodeJS.Timeout;
    
    const shouldPoll = () => {
      // Only poll if:
      // 1. We're in chat tab
      // 2. Task is running or paused
      // 3. Not already polling
      return activeTab === 'chat' && 
             (uiState.isThinking || uiState.continueVisible) && 
             !isPolling;
    };

    const pollState = async () => {
      if (!shouldPoll()) return;

      setIsPolling(true);
      try {
        const lastThought = uiState.agentThoughts.length > 0 
          ? uiState.agentThoughts[uiState.agentThoughts.length - 1].timestamp 
          : '';

        const response = await fetch(`${API_BASE_URL}/get_ui_state?last_thought=${lastThought}`);
        if (response.ok) {
          const newState = await response.json();
          
          // Only update if there are actual changes
          if (JSON.stringify(newState) !== JSON.stringify(uiState)) {
            setUiState(newState);
            setLastUpdateTime(Date.now());
          }
        }
      } catch (error) {
        console.error('Error polling UI state:', error);
      } finally {
        setIsPolling(false);
      }
    };

    if (shouldPoll()) {
      // Initial poll
      pollState();
      
      // Set up interval for subsequent polls
      pollInterval = setInterval(pollState, 2000); // Poll every 2 seconds
    }

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [activeTab, uiState.isThinking, uiState.continueVisible, isPolling, uiState.agentThoughts]); // Added uiState.agentThoughts

  // Update handleSendMessage to reset thoughts properly
  const handleSendMessage = async () => {
    if (!inputMessage.trim() && !filePreview) {
      console.log('[ChatInterface.handleSendMessage] Message not sent: empty message and no file attached');
      return;
    }

    if (!uiState.sendInteractive) {
      console.log('[ChatInterface.handleSendMessage] Message not sent: send not interactive');
      return;
    }

    console.log('[ChatInterface.handleSendMessage] Attempting to send message:', inputMessage);
    setError(null);
    setIsTyping(true);

    // Store the message to send
    const messageToSend = inputMessage;
    const fileToSend = filePreview;
    
    // Clear input and file preview immediately
    setInputMessage('');
    removeFilePreview();

    // Add user message to conversation history immediately
    setUiState(prevState => ({
      ...prevState,
      isThinking: true,
      showThoughts: true,
      agentThoughts: [], // Reset agent thoughts for new message
      conversationHistory: [
        ...prevState.conversationHistory,
        { role: 'user', content: messageToSend }
      ],
      // Preserve local UI state
      isSidebarOpen: prevState.isSidebarOpen,
      isCodeViewOpen: prevState.isCodeViewOpen,
      isCommandPaletteOpen: prevState.isCommandPaletteOpen,
      showSettings: prevState.showSettings
    }));

    try {
      let newState: UIState;
      
      const currentTaskInHistory = uiState.currentTaskId 
        ? uiState.taskHistory.find(s => s.task_id === uiState.currentTaskId)
        : undefined;
        
      const shouldStartNewTask = uiState.currentTaskId === null || 
                                 (currentTaskInHistory !== undefined && 
                                  (currentTaskInHistory.status === 'completed' || currentTaskInHistory.status === 'failed'));

      if (shouldStartNewTask) {
        console.log('[ChatInterface.handleSendMessage] Should start new task based on currentTaskId and status. Creating new task with message:', messageToSend);
        // First create the new task
        newState = await handleNewTask(messageToSend);
        // Then immediately send the message to start the task
        newState = await sendMessage(messageToSend, fileToSend || undefined);
      } else {
        console.log(`[ChatInterface.handleSendMessage] Sending message to current active task (ID: ${uiState.currentTaskId}).`);
        newState = await sendMessage(messageToSend, fileToSend || undefined);
      }

      // Update UI state with the new state while preserving the user's message
      setUiState(prevState => {
        // Create a new conversation history that includes both the user's message and the assistant's response
        const updatedHistory = [...prevState.conversationHistory];
        
        // If there's a new assistant message in the response, add it to the history
        if (newState.conversationHistory.length > 0) {
          const lastMessage = newState.conversationHistory[newState.conversationHistory.length - 1];
          if (lastMessage.role === 'assistant') {
            // Add the assistant's message with its thoughts
            updatedHistory.push({
              ...lastMessage,
              thoughts: newState.agentThoughts || [] // Ensure thoughts is always an array
            });
          }
        }
        
        return {
          ...prevState,
          ...newState,
          showThoughts: true,
          conversationHistory: updatedHistory,
          // agentThoughts: [], // REMOVE THIS LINE - newState.agentThoughts should be used
          // Preserve local UI state
          isSidebarOpen: prevState.isSidebarOpen,
          isCodeViewOpen: prevState.isCodeViewOpen,
          isCommandPaletteOpen: prevState.isCommandPaletteOpen,
          showSettings: prevState.showSettings
        };
      });
      
      if (newState.taskHistory) {
        setTaskSessions(newState.taskHistory);
      }

    } catch (error) {
      console.error('[ChatInterface.handleSendMessage] Error sending message:', error);
      setError(error instanceof Error ? error.message : 'Failed to send message');
    } finally {
      setIsTyping(false);
    }
  };

  // Add effect to log token count changes
  useEffect(() => {
    console.log('[ChatInterface] Token counts updated:', {
      totalTokens: uiState.totalTokens,
      promptTokens: uiState.totalTokens?.prompt_tokens,
      candidatesTokens: uiState.totalTokens?.candidates_tokens,
      totalTokensSum: uiState.totalTokens?.total_tokens
    });
  }, [uiState.totalTokens]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    } else if (e.key === '/' && e.ctrlKey) {
      e.preventDefault();
      setIsCommandPaletteOpen(true);
    }
  };

  const handleStopTask = async () => {
    setError(null);
    try {
      const newState = await stopTask();
      setUiState(newState);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to stop task');
      console.error('Error stopping task:', error);
    }
  };

  const handleContinueTask = async () => {
    setError(null);
    try {
      const newState = await continueTask();
      setUiState(newState);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to continue task');
      console.error('Error continuing task:', error);
    }
  };

  const handleTextareaResize = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        setError('File size exceeds 10MB limit');
        return;
      }
      setFilePreview(file);
    }
  };

  const removeFilePreview = () => {
    setFilePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleCommandPaletteClose = () => {
    setIsCommandPaletteOpen(false);
    setCommandInput('');
  };

  const handleSuggestionClick = async (suggestion: typeof taskSuggestions[0]) => {
    setError(null);
    setIsExecutingTask(true);
    
    try {
      // Start new task
      const response = await fetch(`${API_BASE_URL}/new_task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_name: suggestion.name })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        // Send the task prompt
        const messageResponse = await sendMessage(suggestion.prompt);
        setUiState(messageResponse);
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to execute task');
      console.error('Error executing task:', error);
    } finally {
      setIsExecutingTask(false);
    }
  };

  // Load thoughts for a task
  const loadTaskThoughts = async (taskId: string) => {
    setIsLoadingThoughts(true);
    try {
      const response = await fetch(`${API_BASE_URL}/tasks/thoughts/${taskId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const thoughts = await response.json();
      setSelectedThought(thoughts);
    } catch (error) {
      console.error('Error loading task thoughts:', error);
      setError('Failed to load task thoughts');
    } finally {
      setIsLoadingThoughts(false);
    }
  };

  // Update the renderSidebarContent function to use the new handler
  const renderSidebarContent = () => {
    if (isCodeViewOpen) {
      return <div>Code View Content</div>;
    }
    
    return (
      <div className="flex flex-col h-full">
        <div className="p-4 border-b">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-white">Task History</h2>
            <button
              onClick={() => handleNewTask()}
              className="flex items-center px-3 py-1.5 bg-gradient-to-r from-primary-500 to-secondary-500 text-white rounded-lg hover:from-primary-600 hover:to-secondary-600 transition-colors duration-200"
            >
              <AddIcon className="w-4 h-4 mr-1" />
              New Task
            </button>
          </div>
        </div>
        
        <div className="flex-1 overflow-auto">
          {taskSessions.length === 0 ? (
            <div className="p-4 text-gray-400 text-center">
              No tasks yet. Start a new task to begin.
            </div>
          ) : (
            taskSessions.map(session => (
              <div 
                key={session.task_id} 
                className="p-4 border-b hover:bg-gray-700 cursor-pointer"
                onClick={() => handleTaskHistoryClick(session.task_id)}
              >
                <div className="font-medium text-white">{session.task_name}</div>
                <div className="text-sm text-gray-400">
                  {new Date(session.start_time).toLocaleString()}
                </div>
                <div className="text-sm text-gray-400">
                  Status: {session.status}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    );
  };

  const handleDiagramClick = (diagrams: { code: string; index: number }[]) => {
    const currentTaskId = uiState.currentTaskId;
    if (!currentTaskId) return;

    // If panel is not open, just set the diagrams and open the panel
    if (!isDiagramPanelOpen) {
      setAllDiagrams(diagrams);
      setCurrentTaskDiagrams(prev => ({
        ...prev,
        [currentTaskId]: diagrams
      }));
      setIsDiagramPanelOpen(true);
    } else {
      // If panel is already open, append only new unique diagrams
      setAllDiagrams(prevDiagrams => {
        // Create a Set of existing diagram codes for quick lookup
        const existingCodes = new Set(prevDiagrams.map(d => d.code));
        
        // Filter out diagrams that already exist
        const newDiagrams = diagrams.filter(d => !existingCodes.has(d.code));
        
        // Return combined array only if there are new diagrams
        const updatedDiagrams = newDiagrams.length > 0 ? [...prevDiagrams, ...newDiagrams] : prevDiagrams;
        
        // Update the current task's diagrams
        setCurrentTaskDiagrams(prev => ({
          ...prev,
          [currentTaskId]: updatedDiagrams
        }));
        
        return updatedDiagrams;
      });
    }
  };

  // Add this function to handle API key updates
  const handleApiKeyUpdate = async () => {
    setApiKeyStatus('loading');
    setApiKeyMessage('');

    try {
      const response = await fetch(`${API_BASE_URL}/update_api_key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ api_key: apiKey }),
      });

      const data = await response.json();

      if (response.ok) {
        setApiKeyStatus('success');
        setApiKeyMessage('API key validated and updated successfully!');
        // Clear the input after successful update
        setApiKey('');
      } else {
        setApiKeyStatus('error');
        setApiKeyMessage(data.message || 'Failed to validate API key');
      }
    } catch (err) {
      setApiKeyStatus('error');
      setApiKeyMessage('Failed to update API key. Please try again.');
    }
  };

  // Modify the settings panel render function
  const renderSettingsPanel = () => {
    if (!showSettings) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
        <div className="bg-white rounded-lg p-6 w-full max-w-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Settings</h2>
            <button
              onClick={() => setShowSettings(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              <CloseIcon />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-medium mb-2">API Key Settings</h3>
              <div className="space-y-2">
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter your Gemini API key"
                  className="w-full p-2 border rounded"
                />
                <button
                  onClick={handleApiKeyUpdate}
                  disabled={!apiKey || apiKeyStatus === 'loading'}
                  className={`w-full p-2 rounded text-white ${
                    !apiKey || apiKeyStatus === 'loading'
                      ? 'bg-gray-400'
                      : 'bg-blue-500 hover:bg-blue-600'
                  }`}
                >
                  {apiKeyStatus === 'loading' ? 'Validating...' : 'Update API Key'}
                </button>
                {apiKeyMessage && (
                  <div
                    className={`p-2 rounded ${
                      apiKeyStatus === 'success'
                        ? 'bg-green-100 text-green-700'
                        : apiKeyStatus === 'error'
                        ? 'bg-red-100 text-red-700'
                        : ''
                    }`}
                  >
                    {apiKeyMessage}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Navigation Bar */}
      <div className="bg-gray-800/95 backdrop-blur-sm border-b border-gray-700/50 px-4 py-3 flex justify-between items-center sticky top-0 z-10">
        <div className="flex items-center space-x-4">
          <h1 className="text-lg font-bold text-white">AI Assistant</h1>
          <div className="flex space-x-2">
            <button
              onClick={() => onTabChange('chat')}
              className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === 'chat'
                  ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white shadow-lg shadow-primary-500/20'
                  : 'text-gray-300 hover:bg-gray-700/50'
              }`}
            >
              <ChatIcon className="w-4 h-4 mr-2" />
              Chat
            </button>
            <button
              onClick={() => onTabChange('tasks')}
              className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === 'tasks'
                  ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white shadow-lg shadow-primary-500/20'
                  : 'text-gray-300 hover:bg-gray-700/50'
              }`}
            >
              <ListIcon className="w-4 h-4 mr-2" />
              Tasks
            </button>
            <div className="flex items-center space-x-2 ml-4">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter Gemini API key"
                className="px-3 py-1.5 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-primary-500 focus:outline-none w-64"
              />
              <button
                onClick={handleApiKeyUpdate}
                disabled={!apiKey || apiKeyStatus === 'loading'}
                className={`flex items-center px-3 py-1.5 rounded-lg transition-colors duration-200 ${
                  !apiKey || apiKeyStatus === 'loading'
                    ? 'bg-gray-600 text-gray-400'
                    : 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white hover:from-primary-600 hover:to-secondary-600'
                }`}
              >
                {apiKeyStatus === 'loading' ? 'Validating...' : 'Update Key'}
              </button>
              {apiKeyMessage && (
                <div
                  className={`px-3 py-1.5 rounded-lg text-sm ${
                    apiKeyStatus === 'success'
                      ? 'bg-green-900/50 text-green-400'
                      : apiKeyStatus === 'error'
                      ? 'bg-red-900/50 text-red-400'
                      : ''
                  }`}
                >
                  {apiKeyMessage}
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-300 bg-gray-800/50 px-3 py-1.5 rounded-lg border border-gray-700/50">
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">Tokens:</span>
              <span className="font-mono">{uiState.totalTokens?.total_tokens ?? 0}</span>
              <span className="text-gray-400">
                (P: {uiState.totalTokens?.prompt_tokens ?? 0}, C: {uiState.totalTokens?.candidates_tokens ?? 0})
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <div className={`w-72 bg-gray-800/95 backdrop-blur-sm border-r border-gray-700/50 transition-all duration-300 ${
          isMobile ? (isSidebarOpen ? 'translate-x-0' : '-translate-x-full') : 'translate-x-0'
        }`}>
          {renderSidebarContent()}
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-full">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6" ref={messagesContainerRef}>
            {uiState.conversationHistory.length === 0 && (
              <div className="mb-6">
                <h2 className="text-base font-semibold text-white mb-3">Suggested Tasks</h2>
                <div className="space-y-3">
                  {taskSuggestions.map((suggestion) => (
                    <button
                      key={suggestion.id}
                      onClick={() => handleSuggestionClick(suggestion)}
                      disabled={isExecutingTask}
                      className={`w-full p-4 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors duration-200 text-left ${
                        isExecutingTask ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg bg-gradient-to-r ${suggestion.color}`}>
                            {suggestion.icon}
                          </div>
                          <div>
                            <h3 className="text-sm font-medium text-white">{suggestion.name}</h3>
                            <p className="text-xs text-gray-400 mt-1">{suggestion.description}</p>
                          </div>
                        </div>
                        {isExecutingTask ? (
                          <div className="text-xs text-gray-400">
                            Processing...
                          </div>
                        ) : (
                          <PlayIcon className="w-5 h-5 text-gray-400" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="space-y-4">
              {uiState.conversationHistory.map((message, index) => {
                const isLastMessage = index === uiState.conversationHistory.length - 1;
                const shouldShowThinking = uiState.isThinking && isLastMessage;
                // Keep thoughts visible for the last message if it's from the assistant
                const shouldKeepThoughtsVisible = isLastMessage && message.role === 'assistant';
                
                return (
                  <ChatMessage
                    key={index}
                    message={message}
                    onDiagramClick={handleDiagramClick}
                    isLastMessage={isLastMessage}
                    isThinking={shouldShowThinking}
                    showThoughts={uiState.showThoughts}
                    agentThoughts={uiState.agentThoughts}
                    latestReasoning={uiState.latestReasoning}
                    keepThoughtsVisible={shouldKeepThoughtsVisible}
                  />
                );
              })}

              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-gray-800 rounded-xl px-4 py-3">
                    <button
                      onClick={() => setShowProcessingDetails(!showProcessingDetails)}
                      className="text-sm text-gray-400 hover:text-gray-300 transition-colors duration-200 flex items-center"
                    >
                      <span>Agent is processing...</span>
                      <span className="ml-2">
                        {showProcessingDetails ? '▼' : '►'}
                      </span>
                    </button>
                    {showProcessingDetails && (
                      <div className="mt-2 text-xs text-gray-400 whitespace-pre-wrap max-h-48 overflow-y-auto custom-scrollbar">
                        {uiState.agentThoughts.length > 0 ? (
                          uiState.agentThoughts.map((thought, index) => (
                            <div key={`thought-${index}-${thought.timestamp}`} className="mb-2 pt-2 border-t border-gray-700 first:border-t-0 first:pt-0">
                              <p className="font-medium text-gray-300">{thought.type}:</p>
                              <p className="text-gray-400">{thought.content}</p>
                            </div>
                          ))
                        ) : (
                          <p className="italic">Waiting for agent's first thought...</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Agent Thoughts Modal */}
          {selectedThought && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
              <div className="bg-gray-800 rounded-lg p-4 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-white">Agent Thoughts</h3>
                  <button
                    onClick={() => setSelectedThought(null)}
                    className="p-1 text-gray-400 hover:text-gray-200 transition-colors duration-200"
                  >
                    <CloseIcon className="w-5 h-5" />
                  </button>
                </div>
                <div className="space-y-3">
                  {isLoadingThoughts ? (
                    <div className="flex justify-center">
                      <div className="text-sm text-gray-400">
                        Loading thoughts...
                      </div>
                    </div>
                  ) : (
                    selectedThought.map((thought: any, index: number) => (
                      <div
                        key={index}
                        className="p-3 bg-gray-700 rounded-lg"
                      >
                        <div className="flex justify-between items-start mb-1">
                          <span className="text-xs text-gray-400">
                            {new Date(thought.timestamp).toLocaleTimeString()}
                          </span>
                          <span className="text-xs px-2 py-1 rounded bg-primary-500/20 text-primary-400">
                            {thought.type}
                          </span>
                        </div>
                        <p className="text-sm text-gray-200">{thought.content}</p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Diagram Panel */}
          <MermaidSlidePanel
            isOpen={isDiagramPanelOpen}
            onClose={() => setIsDiagramPanelOpen(false)}
            mermaidCodes={allDiagrams}
          />

          {/* Input Area */}
          <div className="border-t border-gray-700/50 bg-gray-800/95 backdrop-blur-sm p-4">
            <div className="max-w-3xl mx-auto">
              {error && (
                <div className="mb-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <ErrorIcon className="w-4 h-4" />
                    {error}
                  </span>
                  <button
                    onClick={() => setError(null)}
                    className="p-1 hover:text-red-300 transition-colors duration-200"
                  >
                    <CloseIcon className="w-4 h-4" />
                  </button>
                </div>
              )}
              
              {filePreview && (
                <div className="mb-3 p-3 bg-gray-700/50 rounded-lg flex items-center justify-between">
                  <span className="text-gray-300 text-sm flex items-center gap-2">
                    <AttachIcon className="w-4 h-4" />
                    {filePreview.name}
                  </span>
                  <button
                    onClick={() => setFilePreview(null)}
                    className="p-1 text-gray-400 hover:text-gray-200 transition-colors duration-200"
                  >
                    <CloseIcon className="w-4 h-4" />
                  </button>
                </div>
              )}

              <div className="flex items-end gap-3">
                <div className="flex-1 relative">
                  <textarea
                    ref={textareaRef}
                    value={inputMessage}
                    onChange={(e) => {
                      setInputMessage(e.target.value);
                      handleTextareaResize();
                    }}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message..."
                    className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600/50 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-400/50 text-white placeholder-gray-400 resize-none min-h-[44px] max-h-[200px] text-sm overflow-y-auto transition-all duration-200"
                    rows={1}
                    style={{ height: 'auto', minHeight: '44px', maxHeight: '200px' }}
                  />
                  <div className="absolute bottom-3 right-3 text-xs text-gray-400">
                    Press Enter to send
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept=".txt,.pdf,.doc,.docx,.csv,.xlsx,.jpg,.jpeg,.png,.gif"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="p-2 text-gray-400 hover:text-primary-400 transition-colors duration-200 rounded-lg hover:bg-gray-700/50"
                    aria-label="Attach file"
                  >
                    <AttachIcon className="w-5 h-5" />
                  </button>
                  <button
                    onClick={handleSendMessage}
                    disabled={isTyping || (!inputMessage.trim() && !filePreview)}
                    className={`p-2 rounded-lg transition-all duration-200 ${
                      isTyping || (!inputMessage.trim() && !filePreview)
                        ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
                        : 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white hover:from-primary-600 hover:to-secondary-600 shadow-lg shadow-primary-500/20'
                    }`}
                    aria-label="Send message"
                  >
                    <SendIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {renderSettingsPanel()}
    </div>
  );
};

export default ChatInterface; 