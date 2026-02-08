# """
# TODO ATTENTION: THIS FILE WAS ONLY COPIED FROM VLA_PIPELINE; CHECK FOR ERRORS HAS TO BE MADE
# TODO ATTENTION: THIS FILE WAS ONLY AI GENERATED; CHECK FOR ERRORS HAS TO BE MADE

# NetworkAgent: A client agent that forwards action requests to a remote server via ZeroMQ.
# """

# from vla_pipeline.operators.agent import Agent
# import zmq
# import pickle
# import numpy as np
# import sys


# def numpy_to_native(obj, preserve_array_info=False):
#     """Recursively convert NumPy arrays to native Python types for safe serialization.

#     Args:
#         obj: Object to convert
#         preserve_array_info: If True, preserve shape and dtype info for arrays
#     """
#     if isinstance(obj, np.ndarray):
#         if preserve_array_info:
#             return {
#                 "__ndarray__": True,
#                 "data": obj.tolist(),
#                 "dtype": str(obj.dtype),
#                 "shape": obj.shape,
#             }
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {k: numpy_to_native(v, preserve_array_info) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         return type(obj)(numpy_to_native(item, preserve_array_info) for item in obj)
#     elif isinstance(obj, (np.integer, np.floating)):
#         return obj.item()
#     return obj


# def native_to_numpy(obj):
#     """Convert lists back to NumPy arrays where appropriate."""
#     if isinstance(obj, dict):
#         # Check if this is a serialized ndarray with metadata
#         if obj.get("__ndarray__"):
#             dtype = np.dtype(obj["dtype"])
#             return np.array(obj["data"], dtype=dtype).reshape(obj["shape"])
#         return {k: native_to_numpy(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         # Try to convert to numpy array if it looks like numeric data
#         try:
#             arr = np.array(obj)
#             # If it's likely an image (3D array with values 0-255), ensure uint8
#             if (
#                 arr.ndim == 3
#                 and arr.shape[2] in [3, 4]
#                 and arr.min() >= 0
#                 and arr.max() <= 255
#             ):
#                 return arr.astype(np.uint8)
#             return arr
#         except:
#             return [native_to_numpy(item) for item in obj]
#     return obj


# class RemoteAgent(Agent):
#     """
#     NetworkAgent forwards act() calls to a remote agent server via ZeroMQ.

#     The server should be running and listening on the specified address.
#     All robot_state, observation, and instruction data are serialized and sent,
#     and the server returns (action, metadata).
#     """

#     def __init__(self, server_address="tcp://localhost:5555", timeout=30000):
#         """
#         Initialize the NetworkAgent.

#         Args:
#             server_address: ZeroMQ address of the server (e.g., "tcp://localhost:5555")
#             timeout: Timeout in milliseconds for server responses (default: 30000ms = 30s)
#         """
#         super().__init__(
#             action_type=None
#         )  # Action type will be determined by the server

#         self.server_address = server_address
#         self.timeout = timeout

#         # Initialize ZeroMQ context and socket
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.REQ)
#         self.socket.setsockopt(zmq.RCVTIMEO, timeout)
#         self.socket.setsockopt(zmq.SNDTIMEO, timeout)
#         self.socket.connect(server_address)

#         print(f"NetworkAgent connected to server at {server_address}")

#         # Get action type from server
#         self._get_server_info()

#     def _get_server_info(self):
#         """Query the server for its configuration (e.g., action_type)."""
#         try:
#             request = {"type": "info"}
#             self.socket.send(pickle.dumps(request))
#             response = pickle.loads(self.socket.recv())

#             if response.get("status") == "ok":
#                 self.action_type = response.get("action_type", None)
#                 print(f"Server action_type: {self.action_type}")
#             else:
#                 print(
#                     f"Warning: Could not get server info: {response.get('error', 'Unknown error')}"
#                 )
#         except zmq.error.Again:
#             print(
#                 "Warning: Server info request timed out. Continuing without action_type info."
#             )
#             # Recreate socket after timeout
#             self.socket.close()
#             self.socket = self.context.socket(zmq.REQ)
#             self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
#             self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
#             self.socket.connect(self.server_address)
#         except Exception as e:
#             print(f"Warning: Error getting server info: {e}")

#     def act(self, robot_state, observation, instruction=""):
#         """
#         Forward the act request to the remote server.

#         Args:
#             robot_state: Dictionary containing robot state information
#             observation: Dictionary containing sensor observations (e.g., camera images)
#             instruction: Text instruction for the task

#         Returns:
#             Tuple of (action, metadata)
#         """
#         try:
#             # Prepare request - convert NumPy arrays to lists for compatibility
#             # Use preserve_array_info for observations (images) to maintain dtype info
#             request = {
#                 "type": "act",
#                 "robot_state": numpy_to_native(robot_state),
#                 "observation": numpy_to_native(observation, preserve_array_info=True),
#                 "instruction": instruction,
#             }

#             # Send request to server using pickle protocol 4 for compatibility
#             self.socket.send(pickle.dumps(request, protocol=4))

#             # Receive response
#             response = pickle.loads(self.socket.recv())

#             if response.get("status") == "ok":
#                 # Convert lists back to NumPy arrays
#                 action = (
#                     np.array(response["action"])
#                     if isinstance(response["action"], list)
#                     else response["action"]
#                 )

#                 metadata = response.get("metadata", {})

#                 return action, metadata
#             else:
#                 error_msg = response.get("error", "Unknown error")
#                 print(f"Error from server: {error_msg}")
#                 # Return zero action on error
#                 return np.zeros(8), {"error": error_msg}

#         except zmq.error.Again:
#             print(f"Server request timed out after {self.timeout}ms")
#             # Recreate socket after timeout
#             self.socket.close()
#             self.socket = self.context.socket(zmq.REQ)
#             self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
#             self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
#             self.socket.connect(self.server_address)
#             return np.zeros(8), {"error": "timeout"}

#         except Exception as e:
#             print(f"Error communicating with server: {e}")
#             return np.zeros(8), {"error": str(e)}

#     def reset(self):
#         """Send reset signal to the server."""
#         try:
#             request = {"type": "reset"}
#             self.socket.send(pickle.dumps(request))
#             response = pickle.loads(self.socket.recv())

#             if response.get("status") == "ok":
#                 print("Server reset successful")
#             else:
#                 print(f"Server reset failed: {response.get('error', 'Unknown error')}")
#         except Exception as e:
#             print(f"Error resetting server: {e}")

#     def __del__(self):
#         """Clean up ZeroMQ resources."""
#         try:
#             # Send shutdown signal
#             request = {"type": "shutdown"}
#             self.socket.send(pickle.dumps(request))
#             self.socket.recv()  # Wait for acknowledgment
#         except:
#             pass  # Ignore errors during cleanup
#         finally:
#             self.socket.close()
#             self.context.term()
