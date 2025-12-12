/**
 * 	Copyright 2022, Vinícius R. Máximo
 *	Distributed under the terms of the MIT License. 
 *	SPDX-License-Identifier: MIT
 */
package Data;

import java.io.*;

public class File 
{
	PrintWriter outputStream;
	
	public File(String name)
	{
		outputStream = null;
		try {
			outputStream=new PrintWriter(new FileWriter(name)); 
		} 
		catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	public void write(String text) 
	{
		outputStream.println(text);
	}
	
	public void close()
	{
		if (outputStream != null) 
			outputStream.close();
	}
}

