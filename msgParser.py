import re

class MsgParser:
    '''
    A parser for received UDP messages and building UDP messages
    '''
    def __init__(self):
        '''Constructor'''
        pass

    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                if len(items) < 2:
                    print(f"Problem parsing substring: {substr}")
                else:
                    value = [items[i] for i in range(1, len(items))]
                    sensors[items[0]] = value
                b_open = str_sensors.find('(', b_close)
            else:
                print(f"Problem parsing sensor string: {str_sensors}")
                return None
        
        return sensors
    
    def stringify(self, dictionary):
        '''Build a UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value and value[0]:
                msg += f'({key}' + ''.join(f' {str(val)}' for val in value) + ')'
        
        return msg
    
    def parse_race_results(self, result_string):
        '''
        Parse race results from the end-of-race screen
        
        :param result_string: Raw string from TORCS race results
        :return: Dictionary of race results
        '''
        # Regex pattern to match race results
        result_pattern = r'(\d+)\s+(\w+)\s+([:\d.]+)\s+([:\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)'
        
        # Find all matches
        matches = re.findall(result_pattern, result_string)
        
        results = []
        for match in matches:
            result = {
                'rank': int(match[0]),
                'driver': match[1],
                'total_time': match[2],
                'best_lap': match[3],
                'laps': int(match[4]),
                'top_speed': int(match[5]),
                'damages': int(match[6])
            }
            results.append(result)
        
        return results