using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public float smoothness;
    public Transform targetObject;
    private Vector3 initalOffset;
    private Vector3 cameraPosition;

    public enum RelativePosition { InitalPosition, Position1, Position2 }
    public RelativePosition relativePosition;
    public Vector3 position1;
    public Vector3 position2;

    void Start()
    {
        relativePosition = RelativePosition.InitalPosition;
        initalOffset = transform.position - targetObject.position;
    }

    void FixedUpdate()
    {
        cameraPosition = targetObject.position + CameraOffset(relativePosition);
        transform.position = Vector3.Lerp(transform.position, cameraPosition, smoothness * Time.fixedDeltaTime);
        transform.LookAt(targetObject);
    }

    Vector3 CameraOffset(RelativePosition ralativePos)
    {
        Vector3 currentOffset;

        switch (ralativePos)
        {
            case RelativePosition.Position1:
                currentOffset = position1;
                break;

            case RelativePosition.Position2:
                currentOffset = position2;
                break;

            default:
                currentOffset = initalOffset;
                break;
        }
        return currentOffset;
    }
}
