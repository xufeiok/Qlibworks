import yaml
import requests
import datetime

class Notifier:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.notif_config = self.config.get('notification', {})
        self.method = self.notif_config.get('method', 'none')

    def send(self, title: str, message: str):
        """
        Send notification based on configuration.
        """
        full_msg = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}] {title}\n{message}"
        
        if self.method == 'wxauto':
            self._send_wxauto(full_msg)
        elif self.method == 'webhook':
            self._send_webhook(full_msg)
        else:
            print(f"Notification (Mock): {full_msg}")

    def _send_wxauto(self, msg: str):
        try:
            from wxauto import WeChat
            wx = WeChat()
            receiver = self.notif_config.get('wxauto_receiver', 'File Transfer')
            
            # Check if receiver exists in chat list, if not, search?
            # wxauto usually sends to current chat or needs search.
            # Simplified: just send to 'File Transfer' (文件传输助手)
            
            wx.ChatWith(receiver)
            wx.SendMsg(msg)
            print(f"Sent WeChat message to {receiver}")
        except Exception as e:
            print(f"Failed to send WeChat message via wxauto: {e}")
            print("Ensure WeChat PC client is logged in and the receiver is in the chat list.")

    def _send_webhook(self, msg: str):
        url = self.notif_config.get('webhook_url')
        if not url:
            print("Webhook URL not configured.")
            return
        
        try:
            # Simple GET/POST support. Assuming PushPlus/ServerChan style.
            # PushPlus: http://www.pushplus.plus/send?token=XXX&title=XXX&content=XXX
            payload = {'title': 'WeChat Publish Skill', 'content': msg}
            resp = requests.post(url, json=payload) # Try JSON POST first
            if resp.status_code >= 400:
                 requests.get(url, params=payload) # Fallback to GET
            print("Sent webhook notification.")
        except Exception as e:
            print(f"Failed to send webhook: {e}")
