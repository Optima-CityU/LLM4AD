/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Auxiliary;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

/**
 * Utility class for CPU time measurement.
 * Uses ThreadMXBean to get CPU time instead of wall clock time.
 */
public class CpuTime {
	private static ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
	
	/**
	 * Get current CPU time in milliseconds.
	 * @return CPU time in milliseconds
	 */
	public static long getCpuTimeMillis() {
		if (threadBean.isCurrentThreadCpuTimeSupported()) {
			return threadBean.getCurrentThreadCpuTime() / 1_000_000; // Convert nanoseconds to milliseconds
		} else {
			// Fallback to wall clock time if CPU time is not supported
			return System.currentTimeMillis();
		}
	}
	
	/**
	 * Get current CPU time in seconds.
	 * @return CPU time in seconds
	 */
	public static double getCpuTimeSeconds() {
		return getCpuTimeMillis() / 1000.0;
	}
	
	/**
	 * Check if CPU time measurement is supported.
	 * @return true if CPU time is supported, false otherwise
	 */
	public static boolean isSupported() {
		return threadBean.isCurrentThreadCpuTimeSupported();
	}
}

