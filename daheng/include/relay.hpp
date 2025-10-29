#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <iostream>


class Relay {

    public:

        int uart;
        struct termios tty;
        unsigned char* rxbuf;

        Relay(std::string path);

        uint16_t modbus_crc16( const unsigned char *buf, unsigned int len );

        int hexstr_to_bytes(const char *hexstr, unsigned char *out, size_t max_len);

        void send(std::string data);

        void receive();

};