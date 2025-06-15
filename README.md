# PC Automation Agent (Vision-Enabled) 🤖

A powerful AI-powered PC automation agent that combines computer vision, natural language processing, and UI automation to help you automate tasks on your Windows PC. The agent provides intelligent task automation with visual feedback, web integration, and customizable task structures.

## 📸 Visual Interface Showcase

### Main Interface
![Main Interface](images/main_interface.png)
*The intuitive chat-based interface where you interact with the AI agent*

### API Key Configuration
![API Key Setup](images/api_key.png)
*First-time setup: Enter your API key to get started with the agent*

### AI Image Generation
![Image Generation](images/image_generating.png)
*Built-in AI image generation capabilities integrated into your workflows*

### Task Structure Creation
![New Task Structure](images/new_task_structure.png)
*Create new custom task structures for the agent to follow*

### Sequence Diagram Generation
![Sequence Diagram](images/sequence_diagram.png)
*Automatic generation of sequence diagrams for project planning and visualization*

### Task Structure Management
![Task Structure List](images/task_structure_list.png)
*View and manage all your custom task structures in one place*

## 🌟 Features

### 1. Vision-Based Automation
- **Screen Understanding**: The agent can see and understand your screen content in real-time
- **UI Element Detection**: Automatically detects and interacts with UI elements
- **Mouse & Keyboard Control**: Precisely moves the mouse and types based on visual screen analysis
- **Text Recognition**: Built-in OCR capabilities for reading text from images and screens
- **Visual Task Verification**: Confirms task completion through visual feedback

### 2. Natural Language Interaction
- **Conversational Interface**: Chat with the agent in natural language
- **Task Planning**: Agent plans and breaks down complex tasks into steps
- **Context Awareness**: Maintains conversation context and task history
- **Smart Suggestions**: Provides intelligent recommendations and multiple approaches before execution

### 3. Advanced Task Management
- **Custom Task Structures**: Define reusable task templates and workflows
- **Task Templates**: Create structured approaches for repetitive tasks (e.g., "always use Word for reports")
- **Task Customization**: Update, modify, or delete custom task structures as needed
- **Task Pausing/Resuming**: Pause and resume tasks at any point
- **Task History**: View and manage past tasks with detailed logs

### 4. Project Visualization
- **Sequence Diagrams**: Automatically generates sequence diagrams for project approaches
- **Visual Planning**: See the workflow visualization before task execution
- **Process Documentation**: Clear visual representation of task execution steps

### 5. File Operations
- **File Processing**: Read, write, and modify files across different formats
- **File Upload**: Support for multiple file types and batch processing
- **Content Analysis**: Process and analyze file contents intelligently
- **Automated File Organization**: Sort and organize files based on content and metadata

### 6. Web Integration & Content Access
- **Web Search**: Comprehensive web search capabilities
- **Web Navigation**: Automate complex web browsing tasks
- **YouTube Integration**: Access and process YouTube video transcriptions
- **Web Content Extraction**: Extract and process web content from any source

### 7. AI-Powered Image Generation
- **Image Creation**: Generate custom images tailored to task requirements
- **Context-Aware Images**: Creates images that suit the specific task process
- **Image Integration**: Seamlessly incorporates generated images into workflows

### 8. Intelligent Tool Integration
- **Multi-AI Approach**: Leverages existing AI tools when they provide better solutions
- **Tool Recommendation**: Suggests multiple approaches and tools before execution
- **Smart Tool Selection**: Automatically chooses the most suitable tools for each task

### 9. System Automation
- **Application Control**: Launch and control applications with precision
- **System Monitoring**: Monitor system resources and processes
- **Shortcut Management**: Create and manage system shortcuts
- **Process Automation**: Handle complex multi-step system operations

## 🚀 Getting Started

### Prerequisites
- **Windows 10 or later** (Windows-only application)
- **Python 3.8 or higher**
- **Node.js and npm** (for frontend interface)
- **Admin rights** (recommended for full functionality)

### Installation Steps

1. **Clone the Repository**


2. **Create Virtual Environment**
```bash
python -m venv .venv
```

3. **Activate Virtual Environment**
- **Windows (CMD):**
```bash
.venv\Scripts\activate
```
- **Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```
> If activation fails, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

4. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

5. **Install Frontend Dependencies**
```bash
cd frontend
npm install
cd ..
```

## 💻 Usage Examples

### Basic Automation Tasks
```python
# Start the agent
python app_flask.py

# Example commands:
"Open Chrome and search for Python tutorials"
"Take a screenshot and analyze the current window"
"Create a new folder named 'Projects' on my desktop"
"Move the mouse to the search button and click it"
```

### Custom Task Structures
```python
# Define custom task approaches:
"Create a task structure: whenever I ask you to create a report, always use Microsoft Word"
"Update my report task to include a cover page and table of contents"
"Show me all my custom task structures"
"Delete the old presentation task structure"
```

### Advanced Project Management
```python
# Complex project automation:
"Create a new web development project and show me the sequence diagram"
"Plan a data analysis workflow and visualize the process"
"Set up a backup system with visual workflow representation"
```

### Web & Content Integration
```python
# Web and YouTube integration:
"Search YouTube for 'machine learning basics' and get the transcript"
"Find the latest news about AI and create a summary report"
"Download content from this webpage and organize it"
```

### AI-Powered Image Tasks
```python
# Image generation examples:
"Generate a logo for my project and use it in the presentation"
"Create diagrams that explain this concept visually"
"Generate placeholder images for my website mockup"
```

### Intelligent Tool Selection
```python
# Multi-approach examples:
"I need to edit a video - show me all possible approaches"
"Help me create a database - what are the best tools for this?"
"Analyze this dataset and recommend the best analysis method"
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
DEBUG_MODE=false
LOG_LEVEL=INFO
ENABLE_WEB_SEARCH=true
ENABLE_IMAGE_GENERATION=true
```

### Custom Settings
- Modify `config.py` for advanced settings
- Adjust automation parameters in `tools/actions.py`
- Customize UI detection in `vision/vis.py`
- Configure task structures in the frontend interface

## 🛠️ Development

### Project Structure
```
├── agents/           # AI agent implementations
├── tools/           # Utility tools and actions
├── vision/          # Computer vision components
├── task_exec/       # Task execution logic
├── utils/           # Helper utilities
├── chromaDB_management/  # VectoreDB management
├── frontend/        # React-based web interface
└── config.py        # Configuration settings
```

### Adding New Features
1. Create new tool in `tools/` directory
2. Add corresponding agent logic in `agents/`
3. Update frontend components if needed
4. Update configuration files
5. Test thoroughly before deployment

## 🔍 Troubleshooting

### Common Issues
1. **Frontend Not Loading**
   - Ensure you ran `npm install` in the frontend directory
   - Check if Node.js is properly installed

2. **API Key Issues**
   - Verify Google API key in `.env`
   - Check API key permissions for web search and image generation

3. **Permission Errors**
   - Run as administrator for full system access
   - Check file/folder permissions for automation tasks

4. **Vision/Screen Capture Issues**
   - Ensure the application has screen capture permissions
   - Check display scaling settings

### Debug Mode
Enable debug mode in `.env`:
```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

## 📚 API Documentation

### Core Functions
- `execute_action()`: Execute automation actions with visual feedback
- `chat_with_user()`: Handle user interactions and multi-approach suggestions
- `assess_action_outcome()`: Evaluate task results using computer vision
- `generate_sequence_diagram()`: Create visual workflow representations

### Vision Functions
- `capture_full_screen()`: Take and analyze screenshots
- `focus_window_by_title()`: Smart window management
- `move_mouse_and_click()`: Precise mouse control based on screen analysis
- `type_text_at_location()`: Context-aware text input

### Task Management Functions
- `create_task_structure()`: Define custom task templates
- `update_task_structure()`: Modify existing task approaches
- `delete_task_structure()`: Remove obsolete task templates
- `list_task_structures()`: View all custom task configurations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

## 📝 License
This project is licensed under the MIT License - see LICENSE file for details.

## 🚀 Future Enhancements

### Planned Features
- **Enhanced User Interface**: Developing a more intuitive and modern web-based interface
- **Robust Tool Integration Framework**: Advanced system for seamless integration of new AI tools and services
- **Multi-Language Support**: Expanding beyond English for global accessibility
- **Advanced Analytics**: Detailed task performance analytics and optimization suggestions
- **Cross-Platform Support**: Potential expansion to macOS and Linux

### Upcoming Improvements
- **Better Error Handling**: More robust error recovery and user guidance
- **Performance Optimization**: Faster task execution and resource management
- **Enhanced Security**: Improved data protection and secure API handling
- **Plugin System**: Extensible plugin architecture for custom tools

## 🙏 Acknowledgments

Special thanks to the open-source community and the developers of the tools and libraries that make this project possible.

---

**Platform**: Windows Only  
**Version**: Latest  
**Status**: Active Development

**Need Help?** Open an issue or check the documentation for more details!