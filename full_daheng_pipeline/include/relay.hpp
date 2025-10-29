#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <iostream>


#ifndef RELAY.HPP
#define RELAY.HPP




class Request {
    
    public:

        unsigned char* buffer;

        Request(int size);

        Request& address(uint8_t addr);

        Request& read_output(int start, int count);

        Request& write_output(int channel, int value);

        Request& write_multi_output();

        Request& read_input(int start, int count);

        Request& read_spec_reg();

        Request& write_spec_reg();

        Request& write_multi_spec_reg();

        

};


class Relay {

    public:

        int uart;
        struct termios tty;
        unsigned char* rxbuf;

        Relay(std::string path);

        int hexstr_to_bytes(const char *hexstr, unsigned char *out, size_t max_len);

        void send(Request req);

        int receive(uint8_t** buffer);

};
#endif