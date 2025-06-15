import React, { useState, useEffect } from 'react';
import { Task } from '../types';
import { listTasks, saveTask, updateTask, deleteTask } from '../services/api';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Close as CloseIcon,
  Chat as ChatIcon,
  List as ListIcon,
  PlayArrow as PlayIcon,
} from '@mui/icons-material';

interface TaskManagerProps {
  activeTab: 'chat' | 'tasks';
  onTabChange: (tab: 'chat' | 'tasks') => void;
}

const TaskManager: React.FC<TaskManagerProps> = ({ activeTab, onTabChange }) => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingTask, setEditingTask] = useState<Task | null>(null);
  const [taskName, setTaskName] = useState('');
  const [taskPlan, setTaskPlan] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTasks();
  }, []);

  const loadTasks = async () => {
    try {
      const taskList = await listTasks();
      setTasks(taskList);
      setError(null);
    } catch (error) {
      setError('Failed to load tasks');
      console.error('Error loading tasks:', error);
    }
  };

  const handleOpenModal = (task?: Task) => {
    if (task) {
      setEditingTask(task);
      setTaskName(task.name);
      setTaskPlan(task.full_plan);
    } else {
      setEditingTask(null);
      setTaskName('');
      setTaskPlan('');
    }
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setEditingTask(null);
    setTaskName('');
    setTaskPlan('');
    setError(null);
  };

  const handleSaveTask = async () => {
    if (!taskName.trim()) {
      setError('Task name is required');
      return;
    }

    try {
      if (editingTask) {
        await updateTask(taskName, taskPlan);
      } else {
        await saveTask(taskName, taskPlan);
      }
      handleCloseModal();
      loadTasks();
    } catch (error) {
      setError('Failed to save task');
      console.error('Error saving task:', error);
    }
  };

  const handleDeleteTask = async (taskName: string) => {
    if (window.confirm('Are you sure you want to delete this task?')) {
      try {
        await deleteTask(taskName);
        loadTasks();
      } catch (error) {
        setError('Failed to delete task');
        console.error('Error deleting task:', error);
      }
    }
  };

  const handleStartTask = (task: Task) => {
    // Implement task execution logic here
    console.log('Starting task:', task);
  };

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Navigation Bar */}
      <div className="bg-gray-800 border-b border-gray-700 px-3 py-2 flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <h1 className="text-lg font-bold text-white">AI Assistant</h1>
          <div className="flex space-x-1">
            <button
              onClick={() => onTabChange('chat')}
              className={`flex items-center px-2 py-1.5 rounded-md text-sm font-medium transition-colors duration-200 ${
                activeTab === 'chat'
                  ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <ChatIcon className="w-4 h-4 mr-1.5" />
              Chat
            </button>
            <button
              onClick={() => onTabChange('tasks')}
              className={`flex items-center px-2 py-1.5 rounded-md text-sm font-medium transition-colors duration-200 ${
                activeTab === 'tasks'
                  ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <ListIcon className="w-4 h-4 mr-1.5" />
              Tasks
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {/* Saved Tasks */}
        <div>
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-base font-semibold text-white">Saved Tasks</h2>
            <button
              onClick={() => handleOpenModal()}
              className="flex items-center px-3 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-primary-500 to-secondary-500 rounded-lg hover:from-primary-600 hover:to-secondary-600 focus:outline-none focus:ring-2 focus:ring-primary-400 focus:ring-offset-2 focus:ring-offset-gray-800 transition-all duration-200 shadow-lg"
            >
              <AddIcon className="w-4 h-4 mr-1.5" />
              New Task
            </button>
          </div>

          {error && (
            <div className="mb-3 p-2 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-xs flex items-center justify-between">
              <span>{error}</span>
              <button
                onClick={() => setError(null)}
                className="p-1 hover:text-red-300 transition-colors duration-200"
              >
                <CloseIcon className="w-4 h-4" />
              </button>
            </div>
          )}

          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {tasks.map((task) => (
              <div
                key={task.id}
                className="p-4 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors duration-200"
              >
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-sm font-medium text-white">
                    {task.name}
                  </h3>
                  <div className="flex space-x-1">
                    <button
                      onClick={() => handleOpenModal(task)}
                      className="p-1 text-gray-400 hover:text-primary-400 transition-colors duration-200"
                      aria-label="Edit task"
                    >
                      <EditIcon className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeleteTask(task.name)}
                      className="p-1 text-gray-400 hover:text-red-400 transition-colors duration-200"
                      aria-label="Delete task"
                    >
                      <DeleteIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                <p className="text-xs text-gray-300 mb-3">{task.preview}</p>
                <button
                  onClick={() => handleStartTask(task)}
                  className="w-full flex items-center justify-center px-3 py-1.5 text-xs font-medium text-white bg-gradient-to-r from-primary-500 to-secondary-500 rounded-lg hover:from-primary-600 hover:to-secondary-600 transition-all duration-200"
                >
                  <PlayIcon className="w-4 h-4 mr-1" />
                  Start Task
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50 animate-fade-in">
          <div className="bg-gray-800 rounded-lg p-4 w-full max-w-lg border border-gray-700 shadow-2xl">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-base font-semibold text-white">
                {editingTask ? 'Edit Task' : 'New Task'}
              </h3>
              <button
                onClick={handleCloseModal}
                className="p-1 text-gray-400 hover:text-gray-200 transition-colors duration-200"
                aria-label="Close modal"
              >
                <CloseIcon className="w-4 h-4" />
              </button>
            </div>

            {error && (
              <div className="mb-3 p-2 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-xs">
                {error}
              </div>
            )}

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">
                  Task Name
                </label>
                <input
                  type="text"
                  value={taskName}
                  onChange={(e) => setTaskName(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-400 text-white placeholder-gray-400 text-sm"
                  placeholder="Enter task name"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">
                  Task Plan
                </label>
                <textarea
                  value={taskPlan}
                  onChange={(e) => setTaskPlan(e.target.value)}
                  rows={4}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-400 text-white placeholder-gray-400 text-sm resize-none"
                  placeholder="Enter task plan"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={handleCloseModal}
                className="px-3 py-1.5 text-xs font-medium text-gray-300 bg-gray-700 rounded-lg hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 transition-colors duration-200"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveTask}
                className="px-3 py-1.5 text-xs font-medium text-white bg-gradient-to-r from-primary-500 to-secondary-500 rounded-lg hover:from-primary-600 hover:to-secondary-600 focus:outline-none focus:ring-2 focus:ring-primary-400 transition-all duration-200 shadow-lg"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TaskManager; 