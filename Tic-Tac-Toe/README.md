# Tic-Tac-Toe on LAN

    --> If you want to use muliple systems in LAN, then choose a server and put the server system IP address in gameclient.cpp inside the call inet_addr.
    
    --> Compilation (This will generate 2 executables gameserver & gameclient)
            $ g++ -pthread -std=c++17 gameserver.cpp -o gameserver && g++ -pthread -std=c++17 gameclient.cpp -o gameclient

    --> On the server system run
            $ ./gameserver
            
    --> On the client systems run
            $ ./gameclient