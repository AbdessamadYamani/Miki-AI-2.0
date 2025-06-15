import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import TaskManager from './components/TaskManager';
import { Chat as ChatIcon, List as ListIcon } from '@mui/icons-material';
import './styles/mermaid.css';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'tasks'>('chat');

  return (
    <div className="h-screen bg-gray-900 flex flex-col">
      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        {activeTab === 'chat' ? (
          <ChatInterface 
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        ) : (
          <TaskManager 
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        )}
      </main>
    </div>
  );
};

export default App;
