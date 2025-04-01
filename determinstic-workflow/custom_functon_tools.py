from adk import Toolkit, tool
from typing import Dict, Any

class ResearcherToolkit(Toolkit):

    @tool
    def get_trending_keywords(self, topic: str) -> Dict[str, Any]:
        """ Get the trending keywords for a given topic.

        Args:
            topic (str): The topic to get trending keywords for

        Returns:
            str: A JSON string containing the trending keywords
        """

        if topic == "ai":
            return {"keywords": ["agent", "stargate", "openai", "oracle"]}

        elif topic == "tiktok":
            return {"keywords": ["tiktok", "trump", "elon", "larry"]}

        else:
            return {"keywords": ["oracle"]}


class WriterToolkit(Toolkit):

    @tool
    def email_user(self, blog_post: str, user_email: str) -> str:
        """ Email the writer with the blog post.

        Args:
            blog_post (str): The blog post to email to the user
            user_email (str): The email address of the user to email the blog post to
        Returns:
            str: A message indicating that the email has been sent
        """

        return f"Email sent to user with blog post: {blog_post}"
