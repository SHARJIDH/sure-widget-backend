import requests
import uuid
from crewai.tools import tool


STRIPE_MCP_URL = "https://mcp.stripe.com/"

@tool("StripeMCPTool")
def stripe_mcp(name: str, arguments: dict, api_key: str) -> str:
    """
    Calls the Stripe MCP server to execute an action.
    
    Required parameters:
      - api_key (str): Stripe Secret Key. Must be provided in every call. No environment fallback.
    
    Available tools and their parameters:
    
    cancel_subscription:
        Cancel a subscription in Stripe.
        Arguments:
            subscription (str, required): The ID of the subscription to cancel.
        Note: This tool may perform destructive updates and is idempotent.
        
    create_customer:
        Create a customer in Stripe.
        Arguments:
            name (str): The name of the customer.
            email (str, optional): The email of the customer.
            
    create_invoice:
        Create an invoice in Stripe.
        Arguments:
            customer (str): The ID of the customer to create the invoice for.
            days_until_due (int, optional): The number of days until the invoice is due.
            
    create_payment_link:
        Create a payment link in Stripe.
        Arguments:
            price (str): The ID of the price to create the payment link for.
            quantity (int): The quantity of the product to include in the payment link.
            redirect_url (str, optional): The URL to redirect to after the payment is completed.
        Note: This tool interacts with external entities.
        
    create_refund:
        Refund a payment intent in Stripe.
        Arguments:
            payment_intent (str): The ID of the payment intent to refund.
            amount (int, optional): The amount to refund in cents.
            reason (str, optional): The reason for the refund.
        Note: This tool interacts with external entities.
        
    list_customers:
        Fetch a list of Customers from Stripe.
        Arguments:
            limit (int, optional): The number of customers to return.
            email (str, optional): A case-sensitive filter on the list based on the customer's email field.
        Note: This tool is read-only and interacts with external entities.
        
    list_invoices:
        Fetch a list of Invoices from Stripe.
        Arguments:
            customer (str, optional): The ID of the customer to list invoices for.
            limit (int, optional): The number of invoices to return.
        Note: This tool is read-only.
        
    list_prices:
        Fetch a list of Prices from Stripe.
        Arguments:
            product (str, optional): The ID of the product to list prices for.
            limit (int, optional): The number of prices to return.
        Note: This tool is read-only and interacts with external entities.
        
    list_products:
        Fetch a list of Products from Stripe.
        Arguments:
            limit (int, optional): The number of products to return.
        Note: This tool is read-only and interacts with external entities.
        
    list_subscriptions:
        List all subscriptions in Stripe.
        Arguments:
            customer (str, optional): The ID of the customer to list subscriptions for.
            price (str, optional): The ID of the price to list subscriptions for.
            status (str, optional): The status of the subscriptions to list.
            limit (int, optional): The number of subscriptions to return.
        Note: This tool is read-only and interacts with external entities.
        
    search_stripe_documentation:
        Search and retrieve relevant Stripe documentation to answer integration questions.
        Arguments:
            question (str): The user question to search an answer for in the Stripe documentation.
            language (str, optional): The programming language to search for in the documentation.
        Note: This tool is read-only.
        
    update_subscription:
        Update an existing subscription in Stripe.
        Arguments:
            subscription (str, required): The ID of the subscription to update.
            proration_behavior (str, optional): Determines how to handle prorations when the subscription items change.
            items (array, optional): A list of subscription items to update, add, or remove.
                Each item can have:
                - id (str, optional): The ID of the subscription item to modify.
                - price (str, optional): The ID of the price to switch to.
                - quantity (int, optional): The quantity of the plan to subscribe to.
                - deleted (bool, optional): Whether to delete this item.
                
    - name: The MCP tool name (e.g., 'create_customer', 'refund_payment', 'cancel_subscription', 'create_payment_link', 'list_products', 'list_prices', 'list_invoices').
    - arguments: A dictionary with parameters for that tool.
    Returns the JSON-RPC response as a string.
    """
    try:
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
            "id": str(uuid.uuid4())  # unique request ID
        }

        if not api_key:
            raise ValueError("StripeMCPTool requires 'api_key' parameter; no environment fallback.")
        


        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = requests.post(STRIPE_MCP_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Extract and format result if available
        if "result" in data:
            return f"Stripe MCP action '{name}' succeeded.\nResult: {data['result']}"
        elif "error" in data:
            return f"Stripe MCP action '{name}' failed.\nError: {data['error']}"
        else:
            return f"Unexpected response from Stripe MCP: {data}"

    except Exception as e:
        return f"Error calling Stripe MCP '{name}': {str(e)}"
