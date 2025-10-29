#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <bitset>
#include <iostream>

class GPIO {

    public:

        int gpio;

        int number;

        GPIO(int num);

        GPIO() {};

        void init(int num);

        void send(std::string value);

        void get(int* value);

        void trigger();



};