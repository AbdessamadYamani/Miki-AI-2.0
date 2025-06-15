import React, { ReactNode, ComponentPropsWithoutRef, useEffect, useState, useRef, useMemo } from 'react';
import { Message } from '../types';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import MermaidDiagram from './MermaidDiagram';
import MermaidSlidePanel from './MermaidSlidePanel';
import { ExpandMore as ExpandMoreIcon, ExpandLess as ExpandLessIcon, Code as CodeIcon } from '@mui/icons-material';

interface ChatMessageProps {
  message: Message;
  autoOpenMermaid?: boolean;
  onDiagramClick?: (diagrams: { code: string; index: number }[]) => void;
  isLastMessage?: boolean;
  isThinking: boolean;
  showThoughts: boolean;
  latestReasoning?: string | null;
  agentThoughts: Array<{
    timestamp: string;
    content: string;
    type: string;
  }>;
  onReferenceClick?: (url: string) => void;
  keepThoughtsVisible?: boolean;
}

interface CodeProps {
  node?: any;
  inline?: boolean;
  className?: string;
  children?: ReactNode;
  [key: string]: any;
}

// Function to process content and insert references inline
const processContentWithReferences = (content: string, references?: Message['references']) => {
  if (!references || references.length === 0) return content;

  let processedContent = content;
  references.forEach((ref, index) => {
    if (ref.type === 'youtube') {
      const refText = `[${ref.title}](${ref.url})`;
      processedContent = processedContent.replace(refText, `[${ref.title}](${ref.url})`);
    }
  });
  return processedContent;
};

// Function to remove Mermaid code blocks from content
const removeMermaidCodeBlocks = (content: string): string => {
  // Remove explicit markers
  let processedContent = content.replace(/--- START MERMAID CODE ---[\s\S]*?--- END MERMAID CODE ---/g, '');
  
  // Remove markdown code blocks
  processedContent = processedContent.replace(/```mermaid\s*\n[\s\S]*?\n```/g, '');
  
  // Clean up any extra newlines that might be left
  processedContent = processedContent.replace(/\n{3,}/g, '\n\n');
  
  return processedContent.trim();
};

// Improved function to extract all mermaid codes from content
const extractMermaidCodes = (content: string): { code: string; startIndex: number; endIndex: number }[] => {
  console.log('Extracting Mermaid codes from content:', {
    contentLength: content.length,
    hasMarkers: content.includes('--- START MERMAID CODE ---'),
    hasCodeBlocks: content.includes('```mermaid')
  });

  const mermaidCodes: { code: string; startIndex: number; endIndex: number }[] = [];
  
  // Pattern 1: Explicit markers (more reliable)
  const markerPattern = /--- START MERMAID CODE ---([\s\S]*?)--- END MERMAID CODE ---/g;
  let match: RegExpExecArray | null;
  
  while ((match = markerPattern.exec(content)) !== null) {
    const extractedCode = match[1].trim();
    console.log('Found Mermaid code with markers:', {
      startIndex: match.index,
      endIndex: match.index + match[0].length,
      codeLength: extractedCode.length,
      codePreview: extractedCode.substring(0, 100) + (extractedCode.length > 100 ? '...' : ''),
      startsWithValidKeyword: /^(sequenceDiagram|graph|flowchart)/.test(extractedCode)
    });
    
    if (extractedCode.length > 0) {
      mermaidCodes.push({
        code: extractedCode,
        startIndex: match.index,
        endIndex: match.index + match[0].length
      });
    }
  }
  
  // Pattern 2: Mermaid code blocks (fallback)
  const codeBlockPattern = /```mermaid\s*\n([\s\S]*?)\n```/g;
  let codeBlockMatch: RegExpExecArray | null;
  
  while ((codeBlockMatch = codeBlockPattern.exec(content)) !== null) {
    const extractedCode = codeBlockMatch[1].trim();
    console.log('Found Mermaid code in code block:', {
      startIndex: codeBlockMatch.index,
      endIndex: codeBlockMatch.index + codeBlockMatch[0].length,
      codeLength: extractedCode.length,
      codePreview: extractedCode.substring(0, 100) + (extractedCode.length > 100 ? '...' : ''),
      startsWithValidKeyword: /^(sequenceDiagram|graph|flowchart)/.test(extractedCode)
    });
    
    if (extractedCode.length > 0) {
      // Check if this code block overlaps with any marker-based extraction
      const overlaps = mermaidCodes.some(existing => 
        (codeBlockMatch!.index >= existing.startIndex && codeBlockMatch!.index < existing.endIndex) ||
        (existing.startIndex >= codeBlockMatch!.index && existing.startIndex < codeBlockMatch!.index + codeBlockMatch![0].length)
      );
      
      if (!overlaps) {
        mermaidCodes.push({
          code: extractedCode,
          startIndex: codeBlockMatch.index,
          endIndex: codeBlockMatch.index + codeBlockMatch[0].length
        });
      }
    }
  }
  
  // Sort by start index to maintain order
  mermaidCodes.sort((a, b) => a.startIndex - b.startIndex);
  
  console.log('Total unique Mermaid codes found:', mermaidCodes.length);
  mermaidCodes.forEach((code, index) => {
    console.log(`Mermaid code ${index + 1}:`, {
      length: code.code.length,
      preview: code.code.substring(0, 50) + '...',
      validStart: /^(sequenceDiagram|graph|flowchart|gitgraph|pie|journey|gantt|classDiagram|stateDiagram|erDiagram)/.test(code.code.trim())
    });
  });
  
  return mermaidCodes;
};

// Helper function to extract reasoning string from data that might be JSON
const getReasoningStringFromData = (data: string | null | undefined): string | null => {
  if (!data) return null;
  try {
    // Attempt to parse the data as JSON
    const parsed = JSON.parse(data);
    // Check if the parsed object has a 'reasoning' field and it's a string
    if (parsed && typeof parsed.reasoning === 'string') {
      return parsed.reasoning;
    }
    // If it's JSON but not the expected structure, return null or the original string if preferred.
    return null; // For now, only show if 'reasoning' field is found and is a string.
  } catch (e) {
    // If parsing fails, it's not a JSON string. Assume 'data' is already the direct reasoning string.
    return data;
  }
};

// Add this new function to parse JSON thoughts
const parseThoughtContent = (content: string) => {
  try {
    // Try to parse the content as JSON
    const jsonContent = JSON.parse(content);
    
    // Format the thought content
    let formattedContent = '';
    
    if (jsonContent.next_action) {
      formattedContent += `Action: ${jsonContent.next_action.action_type}\n`;
      if (jsonContent.next_action.parameters) {
        formattedContent += `Parameters: ${JSON.stringify(jsonContent.next_action.parameters, null, 2)}\n`;
      }
    }
    
    if (jsonContent.reasoning) {
      formattedContent += `\nReasoning: ${jsonContent.reasoning}`;
    }
    
    return formattedContent;
  } catch (e) {
    // If parsing fails, return the original content
    return content;
  }
};

const CodeBlock: React.FC<CodeProps> = ({ node, inline, className, children, ...props }) => {
  const match = /language-(\w+)/.exec(className || '');
  const codeContent = String(children).replace(/\n$/, '');

  if (match && match[1] === 'mermaid') {
    // Prevent mermaid code blocks from being rendered in the main markdown output
    // The code is already extracted in the parent component.
    return null;
  }

  return !inline && match ? (
    <div className="relative group">
      <div className="absolute -top-2 right-2 px-2 py-1 bg-gray-800 text-gray-300 text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200">
        {match[1]}
      </div>
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={match[1]}
        PreTag="div"
        className="rounded-lg mt-2 shadow-lg"
        {...props}
      >
        {codeContent}
      </SyntaxHighlighter>
    </div>
  ) : (
    <code className={`${className} bg-gray-200 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono`} {...props}>
      {children}
    </code>
  );
};

const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message, 
  autoOpenMermaid = false,
  onDiagramClick,
  isLastMessage = false,
  isThinking,
  showThoughts,
  latestReasoning,
  agentThoughts,
  onReferenceClick,
  keepThoughtsVisible = false
}) => {
  const [showMermaidSlidePanel, setShowMermaidSlidePanel] = useState(false);
  const [selectedDiagramIndex, setSelectedDiagramIndex] = useState(0);
  const [showThoughtsLocal, setShowThoughtsLocal] = useState(keepThoughtsVisible);
  const [showCode, setShowCode] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  useEffect(() => {
    console.log('Message data:', message);
    if (message.image_data) {
      console.log('Image data present:', message.image_data.substring(0, 100) + '...');
      setImageUrl(message.image_data);
    }
  }, [message]);

  // Extract mermaid codes from content
  const mermaidCodes = useMemo(() => extractMermaidCodes(message.content), [message.content]);
  
  // Process content to remove mermaid code blocks
  const processedContent = useMemo(() => {
    console.log('Processing content:', message.content);
    let content = removeMermaidCodeBlocks(message.content);
    content = processContentWithReferences(content, message.references);
    return content;
  }, [message.content, message.references]);

  return (
    <div className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] rounded-lg p-4 ${
        message.role === 'user' 
          ? 'bg-primary-600 text-white' 
          : message.role === 'system'
            ? 'bg-gray-800 text-gray-200'
            : 'bg-gray-800 text-gray-200'
      }`}>
        {/* Message content */}
        <div className="prose prose-sm max-w-none dark:prose-invert" ref={contentRef}>
          <ReactMarkdown
            components={{
              code: CodeBlock,
              pre: ({ children }) => <div>{children}</div>,
            }}
          >
            {processedContent}
          </ReactMarkdown>
          
          {/* Render image if present */}
          {(imageUrl || message.image_data) && (
            <div className="mt-2">
              <img 
                src={imageUrl || message.image_data} 
                alt="Generated content" 
                className="max-w-full rounded-lg shadow-md"
                style={{ maxHeight: '400px', objectFit: 'contain' }}
                onError={(e) => {
                  console.error('Error loading image:', e);
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
          )}

          {/* Mermaid diagrams */}
          {mermaidCodes.length > 0 && (
            <div className="mt-2 space-y-2">
              {mermaidCodes.map((mermaidCode, index) => (
                <div 
                  key={index} 
                  className="border rounded-lg p-2 cursor-pointer"
                  onClick={() => {
                    setSelectedDiagramIndex(index);
                    setShowMermaidSlidePanel(true);
                  }}
                >
                  <MermaidDiagram 
                    chart={mermaidCode.code}
                    id={`mermaid-${index}`}
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Rest of the component remains unchanged */}
        {/* Agent Process Section - Only show for assistant messages */}
        {message.role === 'assistant' && (
          <div className="mt-4 border-t border-gray-700/50 pt-4">
            <button
              onClick={() => setShowThoughtsLocal(!showThoughtsLocal)}
              className="w-full flex items-center justify-between text-left text-sm text-gray-400 hover:text-gray-300 transition-colors duration-200"
              aria-label={showThoughtsLocal ? 'Collapse thoughts' : 'Expand thoughts'}
            >
              <span className="font-medium flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isThinking ? 'bg-blue-500 animate-pulse' : 'bg-gray-500'}`} />
                {isThinking ? 'Agent Processing...' : 'Agent Thoughts'}
              </span>
              {showThoughtsLocal ? (
                <ExpandLessIcon className="w-5 h-5" />
              ) : (
                <ExpandMoreIcon className="w-5 h-5" />
              )}
            </button>
            
            {showThoughtsLocal && (
              <div className="mt-3 space-y-3 bg-gray-800/50 rounded-lg p-3">
                <div className="space-y-2 max-h-60 overflow-y-auto custom-scrollbar">
                  {agentThoughts.map((thought, index) => (
                    <div
                      key={index}
                      className="p-3 rounded-lg bg-gray-700/30 border border-gray-600/30"
                    >
                      <p className="text-xs font-medium text-gray-300 mb-1">{thought.type}</p>
                      <p className="text-sm text-gray-400">{thought.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default React.memo(ChatMessage);