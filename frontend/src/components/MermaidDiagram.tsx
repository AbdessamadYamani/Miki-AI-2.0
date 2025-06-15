import React, { useEffect, useRef, useState, useCallback } from 'react';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
  id?: string;
}

const MermaidDiagram: React.FC<MermaidDiagramProps> = ({ chart, id = 'mermaid-diagram' }) => {
  const elementRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [svgContent, setSvgContent] = useState<string>('');
  const mountedRef = useRef(true);
  const [scale, setScale] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });

  // Initialize mermaid
  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false, // Critical: prevent auto-processing
      theme: 'dark',
      securityLevel: 'loose',
      flowchart: {
        htmlLabels: true,
        useMaxWidth: false,
      },
      sequence: {
        diagramMarginX: 50,
        diagramMarginY: 10,
        actorMargin: 50,
        width: 150,
        height: 65,
        boxMargin: 10,
        boxTextMargin: 5,
        noteMargin: 10,
        messageMargin: 35,
      },
      // Suppress error rendering
      suppressErrorRendering: true,
      logLevel: 'error', // Reduce console noise
    });

    // Clean up any existing mermaid error elements on page
    const cleanupErrors = () => {
      document.querySelectorAll('.mermaid-error-text, .mermaid-error, .mermaid[data-processed="true"]').forEach(el => {
        if (el.textContent?.includes('Syntax error')) {
          el.remove();
        }
      });
    };

    cleanupErrors();
    
    // Set up a mutation observer to catch and remove error elements
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const element = node as Element;
            // Check if it's a mermaid error element
            if (element.classList?.contains('mermaid-error-text') || 
                element.classList?.contains('mermaid-error') ||
                (element.textContent && element.textContent.includes('Syntax error in text'))) {
              element.remove();
            }
            // Also check child elements
            element.querySelectorAll?.('.mermaid-error-text, .mermaid-error').forEach(errorEl => {
              if (errorEl.textContent?.includes('Syntax error')) {
                errorEl.remove();
              }
            });
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    return () => {
      observer.disconnect();
    };
  }, []);

  const cleanChart = useCallback((rawChart: string): string => {
    return rawChart
      .replace(/--- START MERMAID CODE ---/g, '')
      .replace(/--- END MERMAID CODE ---/g, '')
      .replace(/```mermaid\n?/g, '')
      .replace(/```\n?/g, '')
      .trim();
  }, []);

  const renderDiagram = useCallback(async () => {
    if (!mountedRef.current) return;
    
    setIsLoading(true);
    setError(null);
    setSvgContent('');

    // Clear previous diagram and error messages
    if (elementRef.current) {
      elementRef.current.innerHTML = '';
    }

    try {
      const cleanedChart = cleanChart(chart);
      
      if (!cleanedChart) {
        throw new Error('Empty chart content after cleaning');
      }

      // Validate basic mermaid syntax
      const validStarters = [
        'sequenceDiagram',
        'graph',
        'flowchart',
        'gitgraph',
        'pie',
        'journey',
        'gantt',
        'classDiagram',
        'stateDiagram',
        'erDiagram'
      ];

      const isValid = validStarters.some(starter => 
        cleanedChart.toLowerCase().startsWith(starter.toLowerCase())
      );

      if (!isValid) {
        throw new Error(`Invalid Mermaid diagram. Must start with: ${validStarters.join(', ')}`);
      }

      // Generate unique ID for this render
      const uniqueId = `mermaid-render-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      
      let svgResult = null;
      try {
        // Use mermaid.render directly without creating DOM elements
        const result = await mermaid.render(uniqueId, cleanedChart);
        svgResult = result.svg;
      } catch (renderError) {
        console.error('Mermaid render error:', renderError);
        // Clean up any error elements that might have been created
        document.querySelectorAll('.mermaid-error-text, .mermaid-error').forEach(el => el.remove());
        throw renderError;
      }
      
      if (!mountedRef.current) return;
      
      if (svgResult) {
        setSvgContent(svgResult);
        setIsLoading(false);
      } else {
        throw new Error('No SVG returned from mermaid.render');
      }

    } catch (err) {
      console.error('Mermaid rendering error:', err);
      if (!mountedRef.current) return;
      
      // Clear any potential content left in the elementRef on error
      if (elementRef.current) {
        elementRef.current.innerHTML = '';
      }
      
      // Clean up any error elements
      document.querySelectorAll('.mermaid-error-text, .mermaid-error').forEach(el => el.remove());
      
      setError(err instanceof Error ? err.message : 'Unknown rendering error');
      setSvgContent('');
      setIsLoading(false);
    }
  }, [chart, cleanChart]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      // Clean up any error elements when component unmounts
      document.querySelectorAll('.mermaid-error-text, .mermaid-error').forEach(el => {
        if (el.textContent?.includes('Syntax error')) {
          el.remove();
        }
      });
    };
  }, []);

  useEffect(() => {
    renderDiagram();
  }, [renderDiagram]);

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY;
    const zoomFactor = delta > 0 ? 0.9 : 1.1;
    setScale(prevScale => Math.min(Math.max(prevScale * zoomFactor, 0.1), 5));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setStartPos({ x: e.clientX - position.x, y: e.clientY - position.y });
  }, [position]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - startPos.x,
        y: e.clientY - startPos.y
      });
    }
  }, [isDragging, startPos]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const resetZoom = useCallback(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const element = elementRef.current;
    if (element) {
      element.addEventListener('wheel', handleWheel, { passive: false });
      return () => {
        element.removeEventListener('wheel', handleWheel);
      };
    }
  }, [handleWheel]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full min-h-[200px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-400"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full min-h-[200px]">
        <div className="text-red-400 text-center">
          <p className="font-medium">Error rendering diagram:</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-white rounded-lg overflow-hidden p-4 flex items-center justify-center relative">
      <div className="absolute top-4 right-4 z-10 flex space-x-2">
        <button
          onClick={() => setScale(prev => Math.min(prev * 1.2, 5))}
          className="p-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
        </button>
        <button
          onClick={() => setScale(prev => Math.max(prev * 0.8, 0.1))}
          className="p-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
          </svg>
        </button>
        <button
          onClick={resetZoom}
          className="p-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>
      {svgContent && (
        <div 
          className="w-full h-full overflow-hidden"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <div 
            ref={elementRef}
            className="mermaid-svg-container cursor-move"
            dangerouslySetInnerHTML={{ __html: svgContent }}
            style={{
              width: '100%',
              height: 'auto',
              minHeight: '200px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transform: `scale(${scale}) translate(${position.x}px, ${position.y}px)`,
              transformOrigin: 'center',
              transition: isDragging ? 'none' : 'transform 0.1s ease-out',
              userSelect: 'none'
            }}
          />
        </div>
      )}
    </div>
  );
};

export default MermaidDiagram;