5)Variables  
5-0)Basic variables  
데이터 row는 game_id(한 경기에 대한 identifier number), play_id(한 경기에서 선수의 play를 기록한 identifier, play_id와 player(nfl_id으로 구분한다)가 같은 데이터 행에 대해서는 frame_id이 time index 역할을 한다), frame_id (한 플레이에서 time index).
이는 game_id, play_id, nfl_id, frame_id으로 데이터 행을 구분할 수 있으며, game_id, play_id, nfl_id으로 묶인 시계열 데이터는 player의 움직임을 묘사한다.  

5-1)State variables: given variables that affect on the player’s decision (control variables)  
* player_weight: player weight (lbs)  
* player_birth_date: birth date (yyyy-mm-dd)  
* player_position: the player's position (the specific role on the field that they typically play)  
* player_side: team player is on (Offense or Defense)  
* player_role: role player has on play (Defensive Coverage, Targeted Receiver, Passer or Other Route Runner)  
* x: Player position along the long axis of the field, generally within 0 - 120 yards. (numeric)  
* y: Player position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)\  

5-2)Control variables: variables that affect on the player’s state variables  
* s: Speed in yards/second (numeric)  
* a: Acceleration in yards/second^2 (numeric)  
* o: orientation of player (deg)  
* dir: angle of player motion (deg)  

5-3)External variables: Exogenous variables that effect on player  
* ball_land_x: Ball landing position position along the long axis of the field, generally within 0 - 120 yards. (numeric)  
* ball_land_y: Ball landing position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)  
* play_direction: Direction that the offense is moving (left or right)  
* absolute_yardline_number: Distance from end zone for possession team (numeric)  
