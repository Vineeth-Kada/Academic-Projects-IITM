#include <unistd.h>
#include <pthread.h>
#include <iostream>
#include <sys/types.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

char buff[1000];
#define PORT 8000
#define SA struct sockaddr
int socketFD;

// Pass an integer to server
int passInt(string xx, high_resolution_clock::time_point start){
	int x = stoi(xx);
	bzero(buff, sizeof(buff));
	strcpy(buff, to_string(x).c_str());
	
	// If the user gave response after 15 seconds then we don't send it to server.
	// Using 14 instead of 15 because we have to consider the delay of client server communication.
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	if(duration.count() <= 14000) send(socketFD, buff, sizeof(buff), MSG_NOSIGNAL);
	return 1;
}

// Reads string from server and validates it
int readAndValidate(){
	bzero(buff, sizeof(buff));
	read(socketFD, buff, sizeof(buff));
	
	// If servers exits
	if (strncmp(buff, "EXIT", 4) == 0) {
		printf("Exited.\n"); fflush(stdout);
		return 0;
	}
	
	// If patner disconnects
	if (strncmp(buff, "PATDISC", 7) == 0){
		printf("Sorry, your partner disconnected\n"); fflush(stdout);
		return 0;
	}
	
	return 1;
}

// Main function that continuosly communicates with the server
void talkToServer()
{
	int n;
	for (;;) {
		
		if(! readAndValidate()) return;
		printf("%s", buff); fflush(stdout);
		
		// Server is asking whether the player wants to play again.
		if(strncmp(buff, "Do", 2) == 0){
			string x; cin >> x;
			// Repeat until u reach a YES/No
			while(x != "YES" && x != "NO"){
				printf("Please enter YES/NO\n"); fflush(stdout);
				cin >> x;
			}
			bzero(buff, sizeof(buff));
			strcpy(buff, x.c_str());
			send(socketFD, buff, sizeof(buff), MSG_NOSIGNAL);
			if(x == "NO"){
				printf("Bye!\n"); fflush(stdout);
				return;
			}
		}
		// Server is asking the client to enter the row and column
		else if(strncmp(buff, "Enter", 5) == 0){
			high_resolution_clock::time_point start = high_resolution_clock::now();
			for(;;){
				string x, y;
				fflush(stdout); cin >> x >> y;
				if ((x != "1" && x != "2" && x != "3") || (y != "1" && y != "2" && y != "3")){
					printf("Enter values (ROW, COL) in valid range: "); fflush(stdout);
					continue;
				}
				passInt(x, start); passInt(y, start);
				break;
			}
		}
		
	}
}

int main()
{
	struct sockaddr_in serverAddr;

	// Creating socket
	socketFD = socket(AF_INET, SOCK_STREAM, 0);
	if (socketFD == -1) {
		printf("socket creation failed...\n");
		exit(0);
	}
	bzero(&serverAddr, sizeof(serverAddr));

	// assigning IP Address and a PORT
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	serverAddr.sin_port = htons(PORT);

	// connect the client socket to server socket
	if (connect(socketFD, (SA*)&serverAddr, sizeof(serverAddr)) != 0) {
		printf("connection with the server failed...\n");
		exit(0);
	}

	talkToServer();
	close(socketFD);
}
