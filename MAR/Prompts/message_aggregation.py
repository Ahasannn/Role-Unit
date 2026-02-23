from typing import Dict

# ["Normal"]
def message_aggregation(raw_inputs:Dict[str,str], messages:Dict[str,Dict], aggregation_method):
    """
    Aggregate messages from other agents in temporal and spatial dimensions.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
        aggregation_method: str: Aggregation method.
    Returns:
        Any: str: Aggregated message.
    """
    return normal_agg(raw_inputs, messages)

def normal_agg(raw_inputs:Dict[str,str], messages:Dict[str,Dict]):
    """
    Aggregate messages from other agents in temporal and spatial dimensions normally.
    Args:
        messages: Dict[str,Dict]: A dict of messages from other agents.
    Returns:
        Any:str:Aggregated message.
    """
    # Aggregate messages normally
    aggregated_message = ""
    for id, info in messages.items():
        aggregated_message += f"Agent {id}, role is {info['role'].role}, output is:\n\n {info['output']}\n\n"
    return aggregated_message

def inner_test(raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], ):
    return False, ""

