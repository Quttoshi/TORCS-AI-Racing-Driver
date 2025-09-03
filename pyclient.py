import sys
import argparse
import socket
import time
import ai_driver as driver  
import msgParser

def main():
    # Configure the argument parser
    parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

    parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                        help='Host IP address (default: localhost)')
    parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                        help='Host port number (default: 3001)')
    parser.add_argument('--id', action='store', dest='id', default='SCR',
                        help='Bot ID (default: SCR)')
    parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                        help='Maximum number of learning episodes (default: 1)')
    parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                        help='Maximum number of steps (default: 0)')
    parser.add_argument('--track', action='store', dest='track', default=None,
                        help='Name of the track')
    parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
    parser.add_argument('--manual-transmission', action='store_true', dest='manual_transmission',
                        help='Use manual transmission (default: automatic)')

    arguments = parser.parse_args()

    # Print connection summary
    print(f'Connecting to server host IP: {arguments.host_ip} @ port: {arguments.host_port}')
    print(f'Bot ID: {arguments.id}')
    print(f'Maximum episodes: {arguments.max_episodes}')
    print(f'Maximum steps: {arguments.max_steps}')
    print(f'Track: {arguments.track}')
    print(f'Stage: {arguments.stage}')
    transmission_mode = "Manual" if arguments.manual_transmission else "Automatic"
    print(f'Transmission: {transmission_mode}')
    print('*')

    # Create socket with increased timeout
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(30.0)  # Significantly increased timeout
        print("Socket created with 30 second timeout")
    except socket.error as msg:
        print(f'Could not create a socket: {msg}')
        sys.exit(-1)

    # Create message parser and driver
    msg_parser = msgParser.MsgParser()
    
    # Create driver with transmission option
    print("Creating driver instance...")
    auto_transmission = not arguments.manual_transmission
    d = driver.Driver(arguments.stage, auto_transmission=auto_transmission)
    print("Driver instance created")

    # Episode and shutdown control
    shutdownClient = False
    curEpisode = 0
    verbose = False  # Set to True for more detailed output

    try:
        while not shutdownClient:
            # Initialization phase
            connection_attempts = 0
            max_attempts = 10
            
            while connection_attempts < max_attempts:
                connection_attempts += 1
                print(f'Attempt {connection_attempts}/{max_attempts}: Sending ID to server: {arguments.id}')
                
                # Prepare initialization message
                init_message = arguments.id + d.init()
                if verbose:
                    print(f'Sending init string: {init_message[:100]}... (truncated)')
                
                try:
                    sock.sendto(init_message.encode(), (arguments.host_ip, arguments.host_port))
                    if verbose:
                        print("Init message sent successfully")
                except socket.error as send_error:
                    print(f"Failed to send initialization: {send_error}")
                    time.sleep(2)
                    continue
                
                # Wait for response
                try:
                    if verbose:
                        print("Waiting for server response...")
                    buf, addr = sock.recvfrom(1024)
                    response = buf.decode()
                    
                    if verbose:
                        print(f'Received: {response}')
                    
                    if 'identified' in response:
                        print("Successfully connected to server!")
                        connection_attempts = 0  # Reset for next episode
                        break
                
                except socket.timeout:
                    print(f"Connection attempt {connection_attempts} timed out after 30 seconds")
                except Exception as e:
                    print(f"Unexpected error during initialization: {e}")
                
                # Increase delay between attempts
                delay = 2 + (connection_attempts * 0.5)
                if verbose:
                    print(f"Waiting {delay:.1f} seconds before retrying...")
                time.sleep(delay)
            
            # Check if we exceeded max attempts
            if connection_attempts >= max_attempts:
                print(f"Failed to connect after {max_attempts} attempts. Exiting.")
                sys.exit(1)

            # Racing/Simulation phase
            currentStep = 0
            print("Racing phase started")
            
            while True:
                # Receive data from server
                try:
                    buf, addr = sock.recvfrom(1024)
                    buf = buf.decode()
                except socket.timeout:
                    print("Server response timeout. Retrying connection...")
                    break  # Break out and try reconnecting
                except socket.error as e:
                    print(f"Socket error: {e}")
                    break
                
                if verbose:
                    print(f'Received: {buf[:100]}...')

                # Check for shutdown or restart
                if 'shutdown' in buf:
                    d.onShutDown()
                    shutdownClient = True
                    print('\nClient Shutdown')
                    break

                if 'restart' in buf:
                    d.onRestart()
                    print('\nClient Restart')
                    break

                # Increment step counter
                currentStep += 1

                # Drive or end episode
                if currentStep != arguments.max_steps:
                    if buf:
                        try:
                            # Drive and get response
                            buf = d.drive(buf)
                        except Exception as drive_error:
                            print(f"Error in drive function: {drive_error}")
                            buf = '(meta 1)'  # Fallback response
                else:
                    buf = '(meta 1)'  # End of episode

                if verbose:
                    print(f'Sending: {buf[:50]}...')

                # Send data back to server
                if buf:
                    try:
                        sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                    except socket.error as send_error:
                        print(f"Failed to send data: {send_error}")
                        break  # Break out and try reconnecting

            # Update episode counter if we completed an episode
            curEpisode += 1
            if curEpisode >= arguments.max_episodes and arguments.max_episodes > 0:
                shutdownClient = True

    except KeyboardInterrupt:
        print("\nClient interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if hasattr(d, 'onShutDown'):
            d.onShutDown()
        sock.close()
        print("Socket closed.")

if __name__ == '__main__':
    main()