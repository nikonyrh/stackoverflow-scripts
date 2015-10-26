#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <time.h>


// http://www.blackbeltcoder.com/Articles/time/c-high-resolution-timer
#include <Windows.h>

class CHRTimer
{
protected:
    LARGE_INTEGER m_liStart;
    LARGE_INTEGER m_liStop;

public:

    CHRTimer(void)
    {
        m_liStart.QuadPart = m_liStop.QuadPart = 0;
    }

    ~CHRTimer(void)
    {
    }

    // Starts the timer
    void Start()
    {
        ::QueryPerformanceCounter(&m_liStart);
    }

    // Stops the timer
    void Stop()
    {
        ::QueryPerformanceCounter(&m_liStop);
    }

    // Returns the counter at the last Start()
    LONGLONG GetStartCounter()
    {
        return m_liStart.QuadPart;
    }

    // Returns the counter at the last Stop()
    LONGLONG GetStopCounter()
    {
        return m_liStop.QuadPart;
    }

    // Returns the interval between the last Start() and Stop()
    LONGLONG GetElapsed()
    {
        return (m_liStop.QuadPart - m_liStart.QuadPart);
    }

    // Returns the interval between the last Start() and Stop() in seconds
    double GetElapsedAsSeconds()
    {
        LARGE_INTEGER liFrequency;
        ::QueryPerformanceFrequency(&liFrequency);
        return ((double)GetElapsed() / (double)liFrequency.QuadPart);
    }
};

class Utils {
public:
	static double getMillisecondTimestamp() {
		return 1000.0 / CLOCKS_PER_SEC * (double)clock();
	}

	static std::string toBinary(uint64_t v)
	{
		char result[65];
		result[64] = '\0';

		for (int i = 63; i >= 0; --i) {
			result[i] = (v & 1) ? '1' : '0';
			v >>= 1;
		}

		return std::string(result);
	}
};

#endif UTILS_HPP_
