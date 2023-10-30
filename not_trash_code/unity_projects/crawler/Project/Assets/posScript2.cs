using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using UnityEngine;

public class posScript2 : MonoBehaviour
{
    StringLogSideChannel stringLogSideChannel;

    private void Start()
    {
        // Initialize and register the side channel
        stringLogSideChannel = new StringLogSideChannel();
        SideChannelManager.RegisterSideChannel(stringLogSideChannel);
    }

    // Update is called once per frame
    void Update()
    {
        stringLogSideChannel.SendString("agent2 = moved to: " + transform.localPosition.ToString());
    }
}