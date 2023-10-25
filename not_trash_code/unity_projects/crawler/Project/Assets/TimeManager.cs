using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeManager : MonoBehaviour
{
    public float timeScale = 1.0f;  // Default time scale

    void Start()
    {
        Time.timeScale = timeScale; // Set time scale
    }
}
