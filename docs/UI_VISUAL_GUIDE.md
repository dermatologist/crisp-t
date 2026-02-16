# CRISP-T Web UI - Visual Overview

## Main Interface

The Web UI consists of two main panels:

```
┌─────────────────────────────────────────────────────────────────┐
│  🔍 CRISP-T Web UI                                              │
│  Qualitative Research with AI-Powered Analysis                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────────────────────────────────┐
│ Configuration    │ Chat with CRISP-T                            │
│                  │                                              │
│ Model Selection  │ ┌──────────────────────────────────────────┐│
│ ┌──────────────┐ │ │ Welcome to CRISP-T! 👋                  ││
│ │ GPT-5       ▼│ │ │                                          ││
│ └──────────────┘ │ │ I'm your AI-powered qualitative research ││
│                  │ │ assistant. I can help you:               ││
│ Data Source      │ │ • Import and analyze text and numeric    ││
│ ┌──────────────┐ │ │   data                                   ││
│ │ ./data       │ │ │ • Generate coding dictionaries and       ││
│ └──────────────┘ │ │   perform topic modeling                 ││
│                  │ │ • Create visualizations                  ││
│ Advanced Settings│ │ • Link textual findings to numeric       ││
│ ▼ Optional Params│ │   outcomes                               ││
│                  │ │                                          ││
│ ┌──────────────┐ │ └──────────────────────────────────────────┘│
│ │Start Session │ │                                              │
│ └──────────────┘ │ ┌──────────────────────────────────────────┐│
│                  │ │ Type your message here...                ││
│ Status:          │ └──────────────────────────────────────────┘│
│ ● Not connected  │ [Send]                                       │
└──────────────────┴──────────────────────────────────────────────┘
```

## Configuration Panel (Left)

The left panel contains:

1. **Model Selection Dropdown**
   - Dynamically populated with all available models from GitHub Copilot SDK
   - Includes GPT-5.2, GPT-5.1, GPT-5, GPT-4.1, GPT-4o, and more
   - Claude models: Opus 4.6, Opus 4, Sonnet 4.5, Sonnet 4, Sonnet 3.5, Haiku 3.5
   - OpenAI o-series: o3-mini, o1-preview, o1-mini
   - Models are loaded from the API or fallback to comprehensive list
   - Available models depend on your GitHub Copilot subscription

2. **Data Source Input**
   - Text field for path to data files
   - Example: `./data` or `/path/to/research`

3. **Advanced Settings** (Collapsible)
   - Custom Provider checkbox
   - Provider configuration (when checked):
     - Provider Type (OpenAI/Azure/Anthropic)
     - Base URL
     - API Key
   - GitHub Token (optional)

4. **Control Buttons**
   - Start Session (primary button)
   - Stop Session (appears when active)

5. **Status Indicator**
   - Green dot + "Connected (model)" when active
   - Gray dot + "Not connected" when inactive

## Chat Panel (Right)

The right panel contains:

1. **Header**
   - Title: "Chat with CRISP-T"
   - Subtitle: "Ask questions about your qualitative research data"

2. **Messages Area** (Scrollable)
   - Welcome message with examples
   - User messages (right-aligned, blue)
   - Assistant messages (left-aligned, gray)
   - Typing indicator (animated dots)
   - System messages (centered, info style)

3. **Input Area**
   - Multi-line text box for composing messages
   - Send button
   - Keyboard shortcut: Enter to send, Shift+Enter for new line

## Color Scheme

- **Primary Blue**: #0366d6 (buttons, links, user messages)
- **Background**: #f6f8fa (light gray)
- **Panel Background**: #ffffff (white)
- **Text**: #24292e (dark gray)
- **Borders**: #e1e4e8 (light gray)
- **Success**: #28a745 (green - for connected status)
- **Error**: #d73a49 (red - for errors)

## Responsive Design

The interface adapts to different screen sizes:

- **Desktop** (>1024px): Two-column layout as shown above
- **Tablet** (768-1024px): Config panel above chat panel
- **Mobile** (<768px): Stacked layout, optimized for small screens

## Interactive Elements

1. **Start Session Button**
   - Hover: Darker blue
   - Disabled: Gray
   - Shows "Starting..." during initialization

2. **Chat Messages**
   - Fade-in animation when new messages appear
   - Auto-scroll to bottom on new messages
   - Code blocks with syntax highlighting

3. **Typing Indicator**
   - Three animated dots
   - Appears while AI is thinking
   - Removes when response arrives

4. **Configuration**
   - Real-time validation
   - Clear error messages
   - Tooltips for help

## Example Chat Flow

```
┌─────────────────────────────────────────────────────────┐
│ User: Import data from ./interviews                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Assistant: I'll import your data using CRISP-T...       │
│                                                         │
│ Running: crisp --source ./interviews --out corpus       │
│                                                         │
│ ✓ Imported 25 documents                                 │
│ ✓ Found 1 CSV file with 150 rows                        │
│ ✓ Corpus saved to ./corpus                              │
│                                                         │
│ Your data is now ready for analysis!                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ User: Run topic modeling with 5 topics                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Assistant: I'll perform topic modeling on your data...  │
│                                                         │
│ Running: crisp --inp corpus --topics --num 5 --assign  │
│          --out corpus                                   │
│                                                         │
│ Found 5 topics:                                         │
│ 1. Healthcare Access (30% of documents)                 │
│ 2. Cost Concerns (25% of documents)                     │
│ 3. Patient Experience (20% of documents)                │
│ 4. Quality of Care (15% of documents)                   │
│ 5. Technology Use (10% of documents)                    │
│                                                         │
│ Would you like me to create a visualization?            │
└─────────────────────────────────────────────────────────┘
```

## Accessibility

- Semantic HTML structure
- ARIA labels for screen readers
- Keyboard navigation support
- High contrast mode compatible
- Focus indicators on interactive elements

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- Lightweight: ~50KB total (HTML + CSS + JS)
- Message polling: 500ms intervals
- Auto-cleanup of inactive sessions
- Efficient DOM updates

---

For more details, see the full documentation in `docs/ui.md`
