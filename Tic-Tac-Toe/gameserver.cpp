#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <poll.h>
using namespace std;
using namespace std::chrono;

#define PORT 8000
#define SA struct sockaddr

struct thread_args{
	int connFD1;
	int connFD2;
	int id1, id2;
	int gameId;
};

vector<thread_args> global_args;
#define BUFFMAX 1000
#define TIME 15000 // ( in milli seconds )
pthread_mutex_t lockMutex;
int socketFD;
vector<pthread_t> games;
int gameIdCtr = 0;

// These are just declaration, definitions are at a later point
void* playGame(void* args);
int playAgain(char* buff, int* connFD, string log, void* args);

// Creates and new thread & starts the game on that thread
void createGame(struct thread_args args){
	pthread_mutex_lock(&lockMutex);
	args.gameId = gameIdCtr + 1;
	pthread_t tid;
	global_args.push_back(args);
	pthread_create(&tid, NULL, playGame, (void*)&global_args[gameIdCtr]);
	games.push_back(tid);
	gameIdCtr++;
	pthread_mutex_unlock(&lockMutex);
}

// Check the status of the game & return T for Tie, C for continue, 'X' / 'O' based on who won
char ticTacToe(char** state){
	for(int i = 0; i < 3; i++)
		if(state[i][0] != '_' && state[i][0] == state[i][1] && state[i][1] == state[i][2]) return state[i][0];
	
	for(int i = 0; i < 3; i++)
		if(state[0][i] != '_' && state[0][i] == state[1][i] && state[1][i] == state[2][i]) return state[0][i];
	
	if(state[0][0] != '_' && state[0][0] == state[1][1] && state[1][1] == state[2][2]) return state[1][1];
	if(state[2][0] != '-' && state[2][0] == state[1][1] && state[1][1] == state[0][2]) return state[1][1];
	
	bool isTie = true;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			if(state[i][j] == '_') isTie = false;
	
	if(isTie) return 'T';	// Tie
	else return 'C';	// Continue
}

// Sends the current state of the game to the client
void printTicTacToe(char** state, char* buff, int connFD){
	bzero(buff, BUFFMAX);
	string temp = "";
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			temp += state[i][j];
			if(j != 2) temp += " | ";
		}
		temp += "\n";
	}
	temp += "\n\n";
	strcpy(buff, temp.c_str());
	send(connFD, buff, BUFFMAX, MSG_NOSIGNAL);
}

// Sends the message stored in msg to client pointed by the connFD using the auxiliary space pointed by buff.
void sendMsg(char* buff, int connFD, string msg){
	bzero(buff, BUFFMAX);
	strcpy(buff, msg.c_str());
	send(connFD, buff, BUFFMAX, MSG_NOSIGNAL);
}

// Writes the string log to log file
void writeLog(string log){
	// We have to use mutex lock because 2 threads shouldn't write simultaneously to the log file.
	pthread_mutex_lock(&lockMutex);	// Lock
	ofstream logfile;
	logfile.open("log.txt", fstream::app);
	logfile << log << endl;
	logfile.flush();
	logfile.close();
	pthread_mutex_unlock(&lockMutex); // Unlock
	pthread_exit(NULL);	// Close the thread corresponding to the current game.
}

// Receives a string from the client connFD
string recvStr(char* buff, int connFD, int connFDpatner, string log, void* args, int *connFDList, bool move = true){
	bzero(buff, BUFFMAX);
	struct pollfd poll1[] = {{.fd=connFD, .events=POLLIN, .revents=0}};
    int pollReturn = poll(poll1, 1, TIME);	// Using poll to check for time out.
	if(pollReturn > 0){	// No Time Out
		read(connFD, buff, BUFFMAX);
		string temp = buff;
		if(temp == ""){	// Checking if some player disconnected
			sendMsg(buff, connFDpatner, "PATDISC");
			log += "Some Player Disconnected.\n";
			writeLog(log);
		}
		return temp;
	}
	else if(pollReturn == 0){ // Time Out
		sendMsg(buff, connFD, "One/Both of the players haven't responded in 15sec!\n");
		sendMsg(buff, connFDpatner, "One/Both of the players haven't responded in 15sec!\n");
		log += "One/Both of the players haven't responded in 15sec!\n";
		
		if(move){
			log += "Asking players if they want to start a new game.\n";
			if(playAgain(buff, connFDList, log, args)) createGame(* (thread_args*)args);
			else {
				sendMsg(buff, connFDpatner, "EXIT");
				sendMsg(buff, connFD, "EXIT");
			}
		}
		else{
			sendMsg(buff, connFD, "EXIT");
			sendMsg(buff, connFDpatner, "EXIT");
		}
		writeLog(log);
	}
	return to_string(pollReturn);
}

// Receives and integer from the client with FD connFD
int recvInt(char* buff, int connFD, int connFDpatner, string log, void* args, int* connFDList, bool chkTimeOut = false){
	return stoi(recvStr(buff, connFD, connFDpatner, log, args, connFDList, chkTimeOut));
}

// Asks if the players want to play again
int playAgain(char* buff, int* connFD, string log, void* args){
	sendMsg(buff, connFD[0], "Do you want to play the game again (YES/NO)?");
	sendMsg(buff, connFD[1], "Do you want to play the game again (YES/NO)?");
	string player0 = recvStr(buff, connFD[0], connFD[1], log, args, (int*)connFD, false);
	string player1 = recvStr(buff, connFD[1], connFD[0], log, args, (int*)connFD, false);
	if(player0 == "NO" || player1 == "NO"){
		sendMsg(buff, connFD[0], "Sorry, your patner is not available to play another game.\n");
		sendMsg(buff, connFD[1], "Sorry, your patner is not available to play another game.\n");
		return 0;
	}	
	return 1;
}

// Main server function. It continuously listens to the client & replies accordingly
void* playGame(void* args){
	string log = "";
	// Retrieving the arguments that are passed at the thread creation time.
	thread_args* args0 = (thread_args*)args;
	int connFD[] = {args0->connFD1, args0->connFD2};
	int id[] = {args0->id1, args0->id2};
	int gameId = args0->gameId;
	
	char buff[BUFFMAX];
	
	// Contains the state of the ticTacToe game at any current instant.
	char** state;
	state = new char* [3];
	for(int i = 0; i < 3; i++) state[i] = new char[3];
	for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) state[i][j] = '_';
	
	// Send relevent messages to both the players informing the game has started.
	sendMsg(buff, connFD[1], "Connected to the game server. Your player ID is " + to_string(id[1]) + ".\nYour partner's ID is " + to_string(id[0]) + ". Your symbol is 'X'\n\n\nStarting the game...\n_ | _ | _\n_ | _ | _\n_ | _ | _\n\n\n");
	sendMsg(buff, connFD[0], "Your partner's ID is " + to_string(id[1]) + ". Your symbol is 'O'.\n\n\nStarting the game...\n_ | _ | _\n_ | _ | _\n_ | _ | _\n\n\n");
	
	log += "\n\n----------------------------------\n";
	log += "Game#" + to_string(gameId) + " - Players are: " + to_string(id[0]) + ", " + to_string(id[1]) + "\n";
	auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
	log += "Game has stated at Time: " + string(ctime(&timenow)) + "\n";
	
	bool PlayersInterested = true;
	while(PlayersInterested){	// As long as players are interested they will go-on playing the game.
		int ctr = 0;
		bool gameOver = false;
		for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) state[i][j] = '_';
		
		auto start = high_resolution_clock::now();
		while(! gameOver){	// As long as the game is not over we have to repeat this.
			sendMsg(buff, connFD[ctr], "Enter (ROW, COL) for placing your mark: ");
			sendMsg(buff, connFD[(ctr+1)%2], "Waiting for your opponent to make a move.\n");
			
			// Repeat until we reach a valid x, y____________________________________________________________
			int x = recvInt(buff, connFD[ctr], connFD[(ctr+1)%2], log, args, (int*) connFD, true);
			int y = recvInt(buff, connFD[ctr], connFD[(ctr+1)%2], log, args, (int*) connFD, true);
			
			bool validInput = false;
			while(! validInput){
				if(state[x - 1][y - 1] != '_'){
					sendMsg(buff, connFD[ctr], "Enter an unoccupied (ROW, COL) for placing your mark: ");
					x = recvInt(buff, connFD[ctr], connFD[(ctr+1)%2], log, args, (int*) connFD, true);
					y = recvInt(buff, connFD[ctr], connFD[(ctr+1)%2], log, args, (int*) connFD, true);
				}
				else{
					validInput = true;
				}
			}
			//__________________________________________________________________________________________________
			
			log += "Player #" + to_string(id[ctr]) + ": " + to_string(x) + ", " + to_string(y) + "\n"; 
			
			state[x - 1][y - 1] = (ctr & 1 ? 'X' : 'O');	// Update State
			
			// Print Game state in each of the clients
			printTicTacToe(state, buff, connFD[ctr]);
			printTicTacToe(state, buff, connFD[(ctr + 1)%2]);
			
			char result = ticTacToe(state);
			
			gameOver = true;

			// Print the game win/tie/lose in each of the client
			string won = "You won :)\n", lost = "You lost :(\n";
			if(result == 'X'){
				sendMsg(buff, connFD[ctr], (ctr & 1) ? won : lost);
				sendMsg(buff, connFD[(ctr+1)%2], (ctr & 1) ? lost : won);
				log += "player #" + to_string(id[1]) + " has won the game.\n";
			}
			else if(result == 'O'){
				sendMsg(buff, connFD[ctr], (ctr & 1) ? lost : won);
				sendMsg(buff, connFD[(ctr+1)%2], (ctr & 1) ? won : lost);
				log += "player #" + to_string(id[0]) + " has won the game.\n";
			}
			else if(result == 'T'){
				sendMsg(buff, connFD[ctr], "Tie\n");
				sendMsg(buff, connFD[(ctr+1)%2], "Tie\n");
				log += "Game is a tie\n";
			}
			else gameOver = false;
			
			ctr = (ctr + 1)%2;
		}
		
		// Asking players if they are interested to REPLAY.
		log += "Asking players if they want to play again.\n";
		if(! playAgain(buff, (int*)connFD, log, args)) PlayersInterested = false;
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<seconds>(stop - start);
		log += "Time taken by the game(in seconds): " + to_string(duration.count()) + "\n";
		if(PlayersInterested){
			log += "\n[REPLAY]\n\n";
		}
	}
	
	sendMsg(buff, connFD[0], "EXIT");
	sendMsg(buff, connFD[1], "EXIT");
	
	writeLog(log);
	return NULL;
}

int main()
{
	struct sockaddr_in serverAddr;
	
	std::ofstream logFile;
	logFile.open("log.txt", std::ofstream::out | std::ofstream::trunc);
	logFile.close();
	
	// Creating socket
	socketFD = socket(AF_INET, SOCK_STREAM, 0);
	if (socketFD == -1) {
		printf("socket creation failed...\n");
		exit(0);
	}
	bzero(&serverAddr, sizeof(serverAddr));

	// assigning IP Address and a PORT
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	serverAddr.sin_port = htons(PORT);

	// Binding the created socket
	if ((::bind(socketFD, (SA*)&serverAddr, sizeof(serverAddr))) != 0) {
		printf("socket bind failed...\n");
		exit(0);
	}
 
	// Now server is ready to listen
	if ((listen(socketFD, 10)) != 0) {
		printf("Listen failed...\n");
		exit(0);
	}
	else
		printf("Game server started. Waiting for players.\n");

	struct sockaddr_in clientAddr1, clientAddr2;
	unsigned int len = sizeof(clientAddr1);
	
	int ctr = 0;
	while(1){
		thread_args args;
		args.id1 = 2 * ctr + 1;
		args.id2 = 2 * ctr + 2;
		
		args.connFD1 = accept(socketFD, (SA*)&clientAddr1, &len);
		if (args.connFD1 < 0) {
			printf("server accept failed...\n");
			exit(0);
		}
		
		char buff[BUFFMAX];
		sendMsg(buff, args.connFD1, "Connected to the game server. Your player ID is " + to_string(args.id1) + ". Waiting for a partner to join...\n");
		
		args.connFD2 = accept(socketFD, (SA*)&clientAddr2, &len);
		if (args.connFD2 < 0) {
			printf("server accept failed...\n");
			exit(0);
		}
		createGame(args);
		ctr += 1;
	}

	for(int i = 0; i < int(games.size()); i++){
		if(pthread_join(games[i], NULL) != 0)
			printf("Thread join failed\n");
	}
	close(socketFD);
	return 0;
}
