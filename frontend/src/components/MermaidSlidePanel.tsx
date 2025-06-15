import React, { useState } from 'react';
import MermaidDiagram from './MermaidDiagram';
import { Close as CloseIcon, NavigateNext as NextIcon, NavigateBefore as PrevIcon } from '@mui/icons-material';

interface MermaidSlidePanelProps {
  isOpen: boolean;
  onClose: () => void;
  mermaidCodes: { code: string; index: number }[];
}

const MermaidSlidePanel: React.FC<MermaidSlidePanelProps> = ({ isOpen, onClose, mermaidCodes }) => {
  const [activeDiagram, setActiveDiagram] = useState(0);

  // Only reset active diagram when panel is first opened
  React.useEffect(() => {
    if (isOpen && mermaidCodes.length > 0) {
      setActiveDiagram(0);
    }
  }, [isOpen, mermaidCodes.length]);

  const handlePrev = () => {
    setActiveDiagram(prev => (prev > 0 ? prev - 1 : mermaidCodes.length - 1));
  };

  const handleNext = () => {
    setActiveDiagram(prev => (prev < mermaidCodes.length - 1 ? prev + 1 : 0));
  };

  if (!mermaidCodes || mermaidCodes.length === 0) {
    return (
      <div
        className={`fixed top-0 right-0 h-full w-full md:w-1/2 lg:w-1/3 bg-gray-900 shadow-lg transform transition-transform duration-300 ease-in-out z-50 ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col">
          <div className="flex justify-between items-center p-4 border-b border-gray-700">
            <h2 className="text-xl font-semibold text-white">Mermaid Diagrams</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            >
              <CloseIcon className="w-6 h-6 text-gray-400" />
            </button>
          </div>
          <div className="flex-1 flex items-center justify-center">
            <p className="text-gray-400">No diagrams available</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`fixed top-0 right-0 h-full w-full md:w-1/2 lg:w-1/3 bg-gray-900 shadow-lg transform transition-transform duration-300 ease-in-out z-50 ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
    >
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center p-4 border-b border-gray-700">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold text-white">Mermaid Diagrams</h2>
            <span className="text-sm text-gray-400">
              {activeDiagram + 1} of {mermaidCodes.length}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          >
            <CloseIcon className="w-6 h-6 text-gray-400" />
          </button>
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center p-2 border-b border-gray-700">
          <button
            onClick={handlePrev}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            disabled={mermaidCodes.length <= 1}
          >
            <PrevIcon className="w-6 h-6 text-gray-400" />
          </button>
          <button
            onClick={handleNext}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
            disabled={mermaidCodes.length <= 1}
          >
            <NextIcon className="w-6 h-6 text-gray-400" />
          </button>
        </div>

        {/* Diagram Content */}
        <div className="flex-1 overflow-auto p-4">
          <div className="bg-gray-800 rounded-lg p-4 h-full">
            <MermaidDiagram
              chart={mermaidCodes[activeDiagram].code}
              id={`mermaid-diagram-${activeDiagram}`}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MermaidSlidePanel;