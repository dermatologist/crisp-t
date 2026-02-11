// CRISP-T Web UI - Main JavaScript

class CrispUI {
    constructor() {
        this.sessionId = null;
        this.messagePollingInterval = null;
        this.lastMessageCount = 0;
        this.displayedMessageIds = new Set(); // Track which messages have been displayed
        this.isProcessing = false;
        this.lastContentLength = 0; // Track content length to detect stalls
        this.stallCounter = 0; // Count polling cycles with no new content
        this.STALL_THRESHOLD = 3; // Mark complete after 3 polls with no new content (~1.5s)

        this.initializeElements();
        this.attachEventListeners();
        this.checkHealth();
    }

    initializeElements() {
        // Configuration elements
        this.modelSelect = document.getElementById('modelSelect');
        this.dataPath = document.getElementById('dataPath');
        this.useCustomProvider = document.getElementById('useCustomProvider');
        this.providerSettings = document.getElementById('providerSettings');
        this.providerType = document.getElementById('providerType');
        this.providerBaseUrl = document.getElementById('providerBaseUrl');
        this.providerApiKey = document.getElementById('providerApiKey');
        this.githubToken = document.getElementById('githubToken');

        // Control elements
        this.startSessionBtn = document.getElementById('startSession');
        this.stopSessionBtn = document.getElementById('stopSession');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');

        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendMessageBtn = document.getElementById('sendMessage');
    }

    attachEventListeners() {
        this.useCustomProvider.addEventListener('change', () => {
            this.providerSettings.style.display = this.useCustomProvider.checked ? 'block' : 'none';
        });

        this.startSessionBtn.addEventListener('click', () => this.startSession());
        this.stopSessionBtn.addEventListener('click', () => this.stopSession());

        this.sendMessageBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            if (!data.copilot_available) {
                this.showError('Copilot SDK is not installed. Please install with: pip install crisp-t[copilot]');
                this.startSessionBtn.disabled = true;
            }
        } catch (error) {
            this.showError('Unable to connect to server: ' + error.message);
        }
    }

    async startSession() {
        const model = this.modelSelect.value;
        const config = {
            data_path: this.dataPath.value
        };

        // Add GitHub token if provided
        if (this.githubToken.value.trim()) {
            config.github_token = this.githubToken.value.trim();
        }

        // Add custom provider config if enabled
        if (this.useCustomProvider.checked) {
            config.use_custom_provider = true;
            config.provider_type = this.providerType.value;
            config.provider_base_url = this.providerBaseUrl.value.trim();

            if (this.providerApiKey.value.trim()) {
                config.provider_api_key = this.providerApiKey.value.trim();
            }
        }

        try {
            this.startSessionBtn.disabled = true;
            this.startSessionBtn.textContent = 'Starting...';

            // Generate a unique session ID
            this.sessionId = 'session-' + Date.now();

            const response = await fetch('/api/session/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    model: model,
                    config: config
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to create session');
            }

            const data = await response.json();

            // Update UI
            this.updateStatus(true, `Connected (${model})`);
            this.startSessionBtn.style.display = 'none';
            this.stopSessionBtn.style.display = 'block';
            this.chatInput.disabled = false;
            this.sendMessageBtn.disabled = false;

            // Clear welcome message and reset tracking
            this.chatMessages.innerHTML = '';
            this.displayedMessageIds.clear(); // Reset displayed messages tracking
            this.lastMessageCount = 0;
            this.stallCounter = 0;
            this.lastContentLength = 0;

            this.addMessage('assistant', `Hello! I'm your CRISP-T research assistant using ${model}. ` +
                `I'm ready to help you analyze your qualitative research data at ${config.data_path}. ` +
                `What would you like to do?`);

            // Start polling for messages
            this.startMessagePolling();

        } catch (error) {
            this.showError('Failed to start session: ' + error.message);
            this.startSessionBtn.disabled = false;
            this.startSessionBtn.textContent = 'Start Session';
        }
    }

    async stopSession() {
        if (!this.sessionId) return;

        try {
            this.stopMessagePolling();

            const response = await fetch(`/api/session/${this.sessionId}/destroy`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to stop session');
            }

            // Reset UI
            this.sessionId = null;
            this.displayedMessageIds.clear(); // Clear displayed messages
            this.updateStatus(false, 'Not connected');
            this.startSessionBtn.style.display = 'block';
            this.stopSessionBtn.style.display = 'none';
            this.startSessionBtn.disabled = false;
            this.startSessionBtn.textContent = 'Start Session';
            this.chatInput.disabled = true;
            this.sendMessageBtn.disabled = true;

            this.addMessage('system', 'Session ended. Click "Start Session" to begin a new session.');

        } catch (error) {
            this.showError('Failed to stop session: ' + error.message);
        }
    }

    async sendMessage() {
        if (!this.sessionId || this.isProcessing) return;

        const prompt = this.chatInput.value.trim();
        if (!prompt) return;

        try {
            this.isProcessing = true;
            this.chatInput.disabled = true;
            this.sendMessageBtn.disabled = true;
            this.stallCounter = 0; // Reset stall detection
            this.lastContentLength = 0;

            // Add user message to UI
            this.addMessage('user', prompt);
            this.chatInput.value = '';

            // Show typing indicator
            this.showTypingIndicator();

            // Send message to server
            const response = await fetch(`/api/session/${this.sessionId}/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ prompt })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to send message');
            }

            // The response will come through message polling

        } catch (error) {
            this.hideTypingIndicator();
            this.showError('Failed to send message: ' + error.message);
            this.isProcessing = false;
            this.chatInput.disabled = false;
            this.sendMessageBtn.disabled = false;
        }
    }

    startMessagePolling() {
        this.messagePollingInterval = setInterval(() => {
            this.pollMessages();
        }, 500); // Poll every 500ms
    }

    stopMessagePolling() {
        if (this.messagePollingInterval) {
            clearInterval(this.messagePollingInterval);
            this.messagePollingInterval = null;
        }
    }

    async pollMessages() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/session/${this.sessionId}/messages`);

            if (!response.ok) {
                return;
            }

            const data = await response.json();
            const messages = data.messages || [];

            console.log('Polled messages:', messages.length, 'displayed:', this.displayedMessageIds.size);

            // Check if there are new messages since last poll
            if (messages.length > this.lastMessageCount) {
                this.stallCounter = 0; // Reset stall counter on new messages

                // Process new messages starting from last count
                for (let i = this.lastMessageCount; i < messages.length; i++) {
                    const msg = messages[i];
                    const msgId = `${i}-${msg.role}`;

                    if (!this.displayedMessageIds.has(msgId)) {
                        console.log(`New message (index ${i}): ${msg.role}:`, msg.content?.substring(0, 50) + '...');

                        // Add message to UI
                        if (msg.content && msg.content.trim()) {
                            this.addMessage(msg.role, msg.content);
                            this.displayedMessageIds.add(msgId);
                        }
                    }
                }

                this.lastMessageCount = messages.length;
                this.lastContentLength = messages[messages.length - 1]?.content?.length || 0;
            }

            // Check if processing is complete
            if (this.isProcessing && messages.length > 0) {
                const lastMsg = messages[messages.length - 1];

                // Case 1: SESSION_IDLE was received and marked complete
                if (lastMsg && lastMsg.role === 'assistant' && lastMsg.complete === true) {
                    console.log('[DEBUG] Assistant message marked complete by SESSION_IDLE');
                    this.completeResponse();
                    return;
                }

                // Case 2: Detect stall (no new content for multiple polls)
                if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content) {
                    const currentLength = lastMsg.content.length;
                    if (currentLength === this.lastContentLength) {
                        this.stallCounter++;
                        console.log(`[DEBUG] No new content (stall counter: ${this.stallCounter}/${this.STALL_THRESHOLD})`);

                        if (this.stallCounter >= this.STALL_THRESHOLD) {
                            console.log('[DEBUG] Assistant message complete (detected by stall)');
                            this.completeResponse();
                            return;
                        }
                    } else {
                        this.stallCounter = 0; // Reset if we got new content
                        this.lastContentLength = currentLength;
                    }
                }
            }

        } catch (error) {
            console.error('Error polling messages:', error);
        }
    }

    completeResponse() {
        this.hideTypingIndicator();
        this.isProcessing = false;
        this.chatInput.disabled = false;
        this.sendMessageBtn.disabled = false;
        this.stallCounter = 0;
        this.lastContentLength = 0;
    }

    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';

        // Format content (simple markdown-like formatting)
        let formattedContent = this.formatContent(content);
        bubbleDiv.innerHTML = formattedContent;

        messageDiv.appendChild(bubbleDiv);
        this.chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    formatContent(content) {
        // Escape HTML
        const escapeHtml = (text) => {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        };

        // Simple formatting
        let formatted = escapeHtml(content);

        // Code blocks (```...```)
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>');

        // Inline code (`...`)
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold (**...**)
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typing-indicator';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';

        typingDiv.appendChild(indicator);
        this.chatMessages.appendChild(typingDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    updateStatus(connected, text) {
        this.statusIndicator.className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
        this.statusText.textContent = text;
    }

    showError(message) {
        this.addMessage('system', '❌ Error: ' + message);
    }
}

// Initialize the UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.crispUI = new CrispUI();
});
