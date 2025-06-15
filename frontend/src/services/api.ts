import { Message, Task, UIState, ApiResponse } from '../types';

const API_BASE_URL = 'http://localhost:5001';

const handleResponse = async (response: Response) => {
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  
  // Enhanced token count logging
  console.log('[api.handleResponse] Received response data:', {
    hasTotalTokens: !!data.totalTokens,
    totalTokens: data.totalTokens,
    promptTokens: data.totalTokens?.prompt_tokens,
    candidatesTokens: data.totalTokens?.candidates_tokens,
    totalTokensSum: data.totalTokens?.total_tokens,
    responseKeys: Object.keys(data)
  });

  // Ensure totalTokens has the correct structure
  if (data.totalTokens) {
    data.totalTokens = {
      prompt_tokens: data.totalTokens.prompt_tokens ?? 0,
      candidates_tokens: data.totalTokens.candidates_tokens ?? 0,
      total_tokens: data.totalTokens.total_tokens ?? 0
    };
  }

  return data;
};

export const uploadFile = async (file: File): Promise<{ success: boolean; filename: string; filepath: string }> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload_file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
};

export const sendMessage = async (message: string, file?: File): Promise<UIState> => {
  try {
    console.log('[api.sendMessage] Sending message to backend:', message);
    
    let fileData = null;
    if (file) {
      fileData = await uploadFile(file);
    }

    const response = await fetch(`${API_BASE_URL}/send_message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        message,
        file: fileData 
      }),
    });

    console.log('[api.sendMessage] Received response status:', response.status);
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error('[api.sendMessage] Error response from backend:', errorData);
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    // Handle image data in the response
    if (data.image) {
      // Convert the image data to a URL if it's base64
      if (data.image.startsWith('data:image')) {
        data.image = data.image;
      } else {
        // If it's a file path, create a URL to the image
        data.image = `${API_BASE_URL}/images/${data.image}`;
      }
    }
    
    // Log token counts for debugging
    console.log('[api.sendMessage] Successfully parsed response with token counts:', {
      totalTokens: data.totalTokens,
      promptTokens: data.totalTokens?.prompt_tokens,
      candidatesTokens: data.totalTokens?.candidates_tokens,
      totalTokensSum: data.totalTokens?.total_tokens
    });
    return data;
  } catch (error) {
    console.error('[api.sendMessage] Error in sendMessage:', error);
    throw error;
  }
};

export const startNewTask = async (taskName?: string): Promise<UIState> => {
  try {
    const response = await fetch(`${API_BASE_URL}/new_task`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ task_name: taskName || `Task ${Date.now()}` }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error starting new task:', error);
    throw error;
  }
};

export const stopTask = async (): Promise<UIState> => {
  try {
    const response = await fetch(`${API_BASE_URL}/stop_task`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Error stopping task:', error);
    throw error;
  }
};

export const continueTask = async (): Promise<UIState> => {
  try {
    const response = await fetch(`${API_BASE_URL}/continue_task`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Error continuing task:', error);
    throw error;
  }
};

export const listTasks = async (): Promise<Task[]> => {
  const response = await fetch(`${API_BASE_URL}/tasks/list`);
  const data = await handleResponse(response);
  return data.tasks || [];
};

export const saveTask = async (name: string, plan: string): Promise<ApiResponse> => {
  const response = await fetch(`${API_BASE_URL}/tasks/save`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, plan }),
  });
  return handleResponse(response);
};

export const updateTask = async (name: string, plan: string): Promise<ApiResponse> => {
  const response = await fetch(`${API_BASE_URL}/tasks/update`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, plan }),
  });
  return handleResponse(response);
};

export const deleteTask = async (name: string): Promise<ApiResponse> => {
  const response = await fetch(`${API_BASE_URL}/tasks/delete`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name }),
  });
  return handleResponse(response);
};

export const retrieveTask = async (taskId: string): Promise<Task> => {
  const response = await fetch(`${API_BASE_URL}/tasks/retrieve/${taskId}`);
  const data = await handleResponse(response);
  if (!data.task) {
    throw new Error('Task not found');
  }
  return data.task;
};

export const getTaskHistory = async (taskId: string): Promise<any> => {
  try {
    const response = await fetch(`${API_BASE_URL}/tasks/history/${taskId}`);
    if (!response.ok) {
      throw new Error(`Failed to load task history: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error in getTaskHistory:', error);
    throw error;
  }
}; 