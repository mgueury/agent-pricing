from typing import Dict, Any
from adk import Toolkit, tool

class AccountToolkit(Toolkit):

    @tool
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get information about a user by user_id

        Args:
            user_id (str): The user ID to get information about

        Returns:
            Dict[str, Any]: A dictionary containing the user information
        """
        if user_id == "user_123":
            return {
                "user_id": user_id,
                "account_id": "acc_111",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "org_id": "org_222",
            }
        else:
            return {
                "user_id": user_id,
                "account_id": "acc_111",
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "org_id": "org_333",
            }

    @tool
    def get_org_info(self, org_id: str) -> Dict[str, Any]:
        """Get information about an organization by org_id

        Args:
            org_id (str): The organization ID to get information about

        Returns:
            Dict[str, Any]: A dictionary containing the organization information
        """
        return {
            "org_id": org_id,
            "name": "Acme Inc",
            "admin_email": "admin@acme.com",
            "plan": "Enterprise",
        }